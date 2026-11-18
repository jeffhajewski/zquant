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
    printf("zquant %s (built for %s, %s)\n", zq_version(), zq_build_cpu(), zq_build_kernels());
    /* The machine running the tests must be able to run the build under test. */
    if (!zq_cpu_supported()) { printf("FAIL: zq_cpu_supported() is 0 on the build machine\n"); return 1; }

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

    /* Non-finite input is refused, and a refused batch adds nothing. */
    {
        size_t before = zq_index_count(ix);
        float *row = malloc(sizeof(float) * 2 * DIM);
        memcpy(row, base, sizeof(float) * 2 * DIM);
        row[DIM + 3] = NAN;
        if (zq_index_add(ix, row, 2) != ZQ_ERR_INVALID) { printf("FAIL: NaN row accepted\n"); return 1; }
        row[DIM + 3] = INFINITY;
        if (zq_index_add(ix, row, 2) != ZQ_ERR_INVALID) { printf("FAIL: inf row accepted\n"); return 1; }
        if (zq_index_count(ix) != before) { printf("FAIL: rejected batch partly added\n"); return 1; }
        free(row);
    }

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

    {
        float *q = malloc(sizeof(float) * DIM);
        memcpy(q, base, sizeof(float) * DIM);
        q[5] = NAN;
        if (zq_search(ix, se, q, 1, ids, scores) != ZQ_ERR_INVALID) { printf("FAIL: NaN query accepted\n"); return 1; }
        free(q);
    }

    /* A parallel add must give the same index: same results, to the bit. */
    {
        zq_index *par = NULL; zq_searcher *pse = NULL;
        if (zq_index_create(&cfg, &par) != ZQ_OK) { printf("FAIL create par\n"); return 1; }
        if (zq_index_calibrate(par, base, N) != ZQ_OK) { printf("FAIL calibrate par\n"); return 1; }
        if (zq_index_add_parallel(par, base, N, ZQ_MAX_THREADS + 1) != ZQ_ERR_INVALID) { printf("FAIL: thread count above max accepted\n"); return 1; }
        if (zq_index_add_parallel(par, base, N, 4) != ZQ_OK) { printf("FAIL add_parallel\n"); return 1; }
        if (zq_index_count(par) != zq_index_count(ix)) { printf("FAIL: parallel count\n"); return 1; }
        if (zq_searcher_create(par, 32, K, 1, &pse) != ZQ_OK) { printf("FAIL searcher par\n"); return 1; }
        uint32_t *pids = malloc(sizeof(uint32_t) * nq * K);
        float *pscores = malloc(sizeof(float) * nq * K);
        if (zq_search(par, pse, base, nq, pids, pscores) != ZQ_OK) { printf("FAIL search par\n"); return 1; }
        if (memcmp(pids, ids, sizeof(uint32_t) * nq * K) || memcmp(pscores, scores, sizeof(float) * nq * K)) {
            printf("FAIL: parallel add changed search results\n"); return 1;
        }
        free(pids); free(pscores);
        zq_searcher_free(pse); zq_index_free(par);
    }

    /* Save and load: the loaded index answers exactly as the original. */
    {
        size_t size = zq_index_serialized_size(ix), written = 0;
        unsigned char *buf = malloc(size);
        if (zq_index_serialize(ix, buf, size - 1, &written) != ZQ_ERR_INVALID) { printf("FAIL: short buffer accepted\n"); return 1; }
        if (zq_index_serialize(ix, buf, size, &written) != ZQ_OK || written != size) { printf("FAIL serialize\n"); return 1; }
        zq_index *loaded = NULL; zq_searcher *lse = NULL;
        if (zq_index_deserialize(buf, size, &loaded) != ZQ_OK) { printf("FAIL deserialize\n"); return 1; }
        if (zq_index_count(loaded) != zq_index_count(ix)) { printf("FAIL: loaded count\n"); return 1; }
        if (zq_searcher_create(loaded, 32, K, 1, &lse) != ZQ_OK) { printf("FAIL searcher loaded\n"); return 1; }
        uint32_t *lids = malloc(sizeof(uint32_t) * nq * K);
        float *lscores = malloc(sizeof(float) * nq * K);
        if (zq_search(loaded, lse, base, nq, lids, lscores) != ZQ_OK) { printf("FAIL search loaded\n"); return 1; }
        if (memcmp(lids, ids, sizeof(uint32_t) * nq * K) || memcmp(lscores, scores, sizeof(float) * nq * K)) {
            printf("FAIL: loaded index answers differently\n"); return 1;
        }
        /* Damage is reported as a format error. */
        buf[size / 2] ^= 1;
        zq_index *broken = NULL;
        if (zq_index_deserialize(buf, size, &broken) != ZQ_ERR_FORMAT) { printf("FAIL: damaged index accepted\n"); return 1; }
        if (zq_index_deserialize("hello", 5, &broken) != ZQ_ERR_FORMAT) { printf("FAIL: garbage accepted\n"); return 1; }
        free(lids); free(lscores); free(buf);
        zq_searcher_free(lse); zq_index_free(loaded);
    }

    /* zq_score_ids reports exactly the scores search did for the same ids. */
    {
        float got[K];
        if (zq_score_ids(ix, se, base, ids, K, got) != ZQ_OK) { printf("FAIL score_ids\n"); return 1; }
        if (memcmp(got, scores, sizeof got)) { printf("FAIL: score_ids disagrees with search\n"); return 1; }
        uint32_t past = (uint32_t)zq_index_count(ix);
        if (zq_score_ids(ix, se, base, &past, 1, got) != ZQ_ERR_INVALID) { printf("FAIL: out-of-range id scored\n"); return 1; }
    }

    /* Rerank: exact scores against the originals, best first, at most the searcher's k. */
    {
        uint32_t rids[32 * 3]; float rsc[32 * 3];
        if (zq_search_rerank(ix, se, base, nq, base, N, K + 1, rids, rsc) != ZQ_ERR_INVALID) { printf("FAIL: rerank k above searcher k\n"); return 1; }
        if (zq_search_rerank(ix, se, base, nq, base, N - 1, 3, rids, rsc) != ZQ_ERR_INVALID) { printf("FAIL: short originals accepted\n"); return 1; }
        if (zq_search_rerank(ix, se, base, nq, base, N, 3, rids, rsc) != ZQ_OK) { printf("FAIL rerank\n"); return 1; }
        for (size_t i = 0; i < nq; i++) {
            const float *q = base + i * DIM;
            for (int j = 0; j < 3; j++) {
                const float *x = base + (size_t)rids[i * 3 + j] * DIM;
                double dot = 0; for (int t = 0; t < DIM; t++) dot += (double)q[t] * x[t];
                if (fabs(dot - rsc[i * 3 + j]) > 1e-4) { printf("FAIL: rerank score is not exact\n"); return 1; }
                if (j && rsc[i * 3 + j] > rsc[i * 3 + j - 1]) { printf("FAIL: rerank order\n"); return 1; }
            }
            if (rids[i * 3] != (uint32_t)i) { printf("FAIL: rerank lost the self match\n"); return 1; }
        }
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
    {
        float saved = base[3];
        base[3] = NAN;
        if (zq_codec_encode(cd, base, 1, cbuf, nbuf) != ZQ_ERR_INVALID) { printf("FAIL: codec NaN accepted\n"); return 1; }
        base[3] = saved;
    }
    zq_codec_free(cd);
    free(cbuf); free(nbuf); free(back);

    zq_searcher_free(se);
    zq_index_free(ix);
    free(base); free(ids); free(scores);
    printf("PASS\n");
    return 0;
}
