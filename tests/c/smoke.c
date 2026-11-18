/* Exercises the C ABI the way a binding author would: create, calibrate, add, search,
 * and check that a vector in the corpus retrieves itself. Also checks the error paths,
 * since a binding's first encounter with the library is usually a mistake. */
#include "zquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DIM 64
#define N   2000
#define K   10

static unsigned long long s = 88172645463325252ULL;
static double rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (double)(s >> 11) / 9007199254740992.0; }
static float gauss(void) { double u = rnd(), v = rnd(); return (float)(sqrt(-2*log(u+1e-12))*cos(6.283185307*v)); }

int main(void) {
    printf("zquant %s\n", zq_version());

    float *base = malloc(sizeof(float) * N * DIM);
    for (int i = 0; i < N; i++) {
        double sq = 0;
        for (int j = 0; j < DIM; j++) { base[i*DIM+j] = gauss(); sq += base[i*DIM+j]*base[i*DIM+j]; }
        float inv = (float)(1.0/sqrt(sq));
        for (int j = 0; j < DIM; j++) base[i*DIM+j] *= inv;
    }

    zq_config cfg = { .dim = DIM, .bits = 5, .metric = ZQ_METRIC_INNER_PRODUCT, .seed = 42, .compact = 1 };
    zq_index *ix = NULL;
    zq_status st = zq_index_create(&cfg, &ix);
    if (st != ZQ_OK) { printf("FAIL create: %s\n", zq_status_string(st)); return 1; }

    st = zq_index_calibrate(ix, base, N);
    if (st != ZQ_OK) { printf("FAIL calibrate: %s\n", zq_status_string(st)); return 1; }
    st = zq_index_add(ix, base, N);
    if (st != ZQ_OK) { printf("FAIL add: %s\n", zq_status_string(st)); return 1; }

    printf("count=%zu  dim=%u  bytes/vector=%zu\n",
           zq_index_count(ix), zq_index_dim(ix), zq_index_bytes_per_vector(ix));

    /* calibrate after add must be refused, not crash */
    if (zq_index_calibrate(ix, base, N) != ZQ_ERR_STATE) { printf("FAIL: calibrate after add allowed\n"); return 1; }
    /* null arguments must be refused */
    if (zq_index_create(NULL, &ix) != ZQ_ERR_INVALID) { printf("FAIL: null config accepted\n"); return 1; }
    if (zq_search(ix, NULL, base, 1, NULL, NULL) != ZQ_ERR_INVALID) { printf("FAIL: null searcher accepted\n"); return 1; }
    /* out-of-range bits must be refused */
    zq_config bad = cfg; bad.bits = 99; zq_index *tmp = NULL;
    if (zq_index_create(&bad, &tmp) != ZQ_ERR_INVALID) { printf("FAIL: bits=99 accepted\n"); return 1; }
    /* bits=1 leaves the index's codebook no bits; it used to be accepted and crash later */
    bad.bits = 1;
    if (zq_index_create(&bad, &tmp) != ZQ_ERR_INVALID) { printf("FAIL: bits=1 accepted\n"); return 1; }
    /* a dimension past the limit, including one that would overflow when rounded up */
    bad = cfg; bad.dim = ZQ_MAX_DIM + 1;
    if (zq_index_create(&bad, &tmp) != ZQ_ERR_INVALID) { printf("FAIL: dim above max accepted\n"); return 1; }
    bad.dim = 0xFFFFFFFFu;
    if (zq_index_create(&bad, &tmp) != ZQ_ERR_INVALID) { printf("FAIL: dim=2^32-1 accepted\n"); return 1; }
    /* sizes whose product wraps must be refused, not turned into a small allocation */
    zq_searcher *huge = NULL;
    if (zq_searcher_create(ix, 32, (size_t)1 << 59, 1, &huge) != ZQ_ERR_INVALID) { printf("FAIL: wrapping k accepted\n"); return 1; }
    if (zq_searcher_create(ix, 32, K, ZQ_MAX_THREADS + 1, &huge) != ZQ_ERR_INVALID) { printf("FAIL: thread count above max accepted\n"); return 1; }
    if (zq_index_add(ix, base, ((size_t)-1) / 2) != ZQ_ERR_INVALID) { printf("FAIL: wrapping n accepted\n"); return 1; }
    zq_codec_config cbad = { .dim = 0xFFFFFFFFu, .bits = 4, .seed = 1 }; zq_codec *ctmp = NULL;
    if (zq_codec_create(&cbad, &ctmp) != ZQ_ERR_INVALID) { printf("FAIL: codec dim=2^32-1 accepted\n"); return 1; }

    zq_searcher *se = NULL;
    st = zq_searcher_create(ix, 32, K, 1, &se);
    if (st != ZQ_OK) { printf("FAIL searcher: %s\n", zq_status_string(st)); return 1; }

    const size_t nq = 32;
    uint32_t *ids = malloc(sizeof(uint32_t) * nq * K);
    float *scores = malloc(sizeof(float) * nq * K);
    st = zq_search(ix, se, base, nq, ids, scores);
    if (st != ZQ_OK) { printf("FAIL search: %s\n", zq_status_string(st)); return 1; }

    /* A searcher belongs to the index it was created from. Used on another - here one
     * with a much larger dimension - it would index past its own buffers. */
    {
        zq_config big = cfg; big.dim = 1024; zq_index *other = NULL;
        if (zq_index_create(&big, &other) != ZQ_OK) { printf("FAIL create other\n"); return 1; }
        float *q = calloc(1024, sizeof(float)); q[0] = 1;
        if (zq_index_add(other, q, 1) != ZQ_OK) { printf("FAIL add other\n"); return 1; }
        if (zq_search(other, se, q, 1, ids, scores) != ZQ_ERR_INVALID) { printf("FAIL: foreign searcher accepted\n"); return 1; }
        free(q);
        zq_index_free(other);
    }

    /* Each corpus vector queried against itself should rank itself first. */
    int self_hits = 0;
    for (size_t i = 0; i < nq; i++) if (ids[i*K] == (uint32_t)i) self_hits++;
    printf("self-retrieval: %d/%zu  top score %.4f\n", self_hits, nq, scores[0]);
    if (self_hits < (int)nq - 1) { printf("FAIL: self-retrieval too low\n"); return 1; }

    /* Codec round-trip: the KV-cache path. Checks that decoding reconstructs the input
     * to within quantization error, and that the packed size is what the header says. */
    zq_codec_config ccfg = { .dim = DIM, .bits = 5, .seed = 7 };
    zq_codec *cd = NULL;
    if (zq_codec_create(&ccfg, &cd) != ZQ_OK) { printf("FAIL codec create\n"); return 1; }
    size_t cb = zq_codec_code_bytes(cd);
    printf("codec: %zu code bytes/vector (%d floats = %zu bytes raw)\n", cb, DIM, sizeof(float)*DIM);
    if (cb == 0 || cb > sizeof(float) * DIM) { printf("FAIL: implausible code size\n"); return 1; }

    const size_t rows = 256;
    unsigned char *cbuf = malloc(cb * rows);
    float *nbuf = malloc(sizeof(float) * rows);
    float *back = malloc(sizeof(float) * rows * DIM);
    if (zq_codec_encode(cd, base, rows, cbuf, nbuf) != ZQ_OK) { printf("FAIL encode\n"); return 1; }
    if (zq_codec_decode(cd, cbuf, nbuf, rows, back) != ZQ_OK) { printf("FAIL decode\n"); return 1; }

    double num = 0, den = 0;
    for (size_t i = 0; i < rows * DIM; i++) {
        double e = base[i] - back[i];
        num += e * e; den += (double)base[i] * base[i];
    }
    double rel = sqrt(num / den);
    printf("codec round-trip relative error: %.4f\n", rel);
    if (!(rel < 0.25)) { printf("FAIL: reconstruction error too large\n"); return 1; }
    if (zq_codec_encode(cd, NULL, rows, cbuf, nbuf) != ZQ_ERR_INVALID) { printf("FAIL: null rows accepted\n"); return 1; }
    zq_codec_free(cd);
    free(cbuf); free(nbuf); free(back);

    zq_searcher_free(se);
    zq_index_free(ix);
    free(base); free(ids); free(scores);
    printf("PASS\n");
    return 0;
}
