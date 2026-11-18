/* zquant — TurboQuant vector quantization. C ABI.
 *
 * Handles are opaque; nothing here depends on the library's internal layout. Every
 * fallible call returns zq_status rather than aborting, because a language binding
 * cannot recover from a panic.
 *
 * Threading: an index is safe to search from multiple threads provided each thread owns
 * its own zq_searcher. Mutating calls (add, calibrate) are not concurrent-safe.
 */
#ifndef ZQUANT_H
#define ZQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int zq_status;
#define ZQ_OK               0
#define ZQ_ERR_ALLOC       -1
#define ZQ_ERR_INVALID     -2
#define ZQ_ERR_STATE       -3
#define ZQ_ERR_UNSUPPORTED -4
#define ZQ_ERR_CPU         -5
#define ZQ_ERR_FORMAT      -6

/* Largest accepted dimension, and largest worker count for a searcher. */
#define ZQ_MAX_DIM     65536
#define ZQ_MAX_THREADS 1024

#define ZQ_METRIC_INNER_PRODUCT 0
#define ZQ_METRIC_COSINE        1
#define ZQ_METRIC_L2            2

typedef struct zq_index zq_index;
typedef struct zq_searcher zq_searcher;

typedef struct {
    /* 1..ZQ_MAX_DIM. */
    uint32_t dim;
    /* Total bit budget per coordinate, 2..6. The scalar codebook uses bits-1, so 1 is
     * rejected rather than leaving it no bits. */
    uint8_t  bits;
    int      metric;
    uint64_t seed;
    /* Non-zero keeps codes packed (smaller); zero dequantizes to int8 (faster, ~2x memory). */
    int      compact;
} zq_config;

const char *zq_version(void);
const char *zq_status_string(zq_status status);

/* Non-zero if this CPU can run this build of the library.
 *
 * Release builds use instructions beyond the architecture baseline (SSSE3 on x86_64,
 * the dot-product extension on arm64), because without them search is 6-10x slower.
 * On a CPU that lacks them the library would die of an illegal instruction, so call
 * this first, straight after loading it: it is compiled for the bare baseline and is
 * safe on any CPU. If it returns 0, use a portable build instead. Creating an index or
 * codec also checks, returning ZQ_ERR_CPU. */
int zq_cpu_supported(void);

/* The CPU model this build was compiled for, e.g. "x86_64_v2", for diagnostics. */
const char *zq_build_cpu(void);

/* Which scan kernels searches use, e.g. "zquant-kernels shuffle=native dot=native".
 * "scalar" means the portable fallback, which is correct but several times slower. An
 * x86_64 build below AVX2 also carries kernels compiled for AVX2 and uses them on a CPU
 * that has it, reporting "... wide=avx2"; the scores are identical either way. Setting
 * ZQUANT_KERNELS=none in the environment before loading the library keeps it to its
 * own kernels. */
const char *zq_build_kernels(void);

zq_status zq_index_create(const zq_config *config, zq_index **out);
void      zq_index_free(zq_index *index);

/* Fit per-coordinate shift and scale from a sample. Must precede any add.
 *
 * Worth doing when the corpus centroid sits away from the origin: compute
 * ||mean(x/||x||)|| on a sample and expect a gain above roughly 0.3, and none below.
 * It is not free - on low-rank zero-mean data it can cost recall. */
zq_status zq_index_calibrate(zq_index *index, const float *rows, size_t n);

/* Measure the sample's centroid norm and calibrate only if it is at least 0.3, where
 * calibration measured a gain; below it, it measured neutral or worse. Stores the norm
 * in *centroid_norm and whether calibration was applied in *applied; either may be
 * NULL. Fails exactly as zq_index_calibrate would, whatever the measurement. The
 * sensible default when you do not know your data. */
zq_status zq_index_calibrate_auto(zq_index *index, const float *rows, size_t n,
                                  float *centroid_norm, int *applied);

/* Append n row-major vectors of dim floats each. A batch containing any NaN or
 * infinity is refused whole with ZQ_ERR_INVALID; nothing from it is added. Queries,
 * calibration samples and codec input are checked the same way. */
zq_status zq_index_add(zq_index *index, const float *rows, size_t n);

/* zq_index_add, encoding on up to `threads` threads (0 or 1: the calling thread;
 * at most ZQ_MAX_THREADS). The index ends up byte-for-byte the same as after
 * zq_index_add. Below a few hundred vectors per thread it runs on the calling thread,
 * since starting a thread costs more than that much encoding. Like every call that
 * modifies an index, it must not overlap any other call on the same index. */
zq_status zq_index_add_parallel(zq_index *index, const float *rows, size_t n, size_t threads);

/* Saving and loading.
 *
 * The serialized form holds the configuration, any calibration and the codes, in a
 * versioned, checksummed, little-endian format that is the same on every platform.
 * Loading rebuilds the rest from the configuration, so a loaded index searches and
 * accepts new vectors exactly as the saved one did.
 *
 * zq_index_serialize needs a buffer of at least zq_index_serialized_size bytes
 * (ZQ_ERR_INVALID otherwise) and stores the bytes written in *written, which may be
 * NULL. zq_index_deserialize copies what it needs, so the buffer can be freed
 * afterwards. It returns ZQ_ERR_FORMAT for anything that is not an intact index
 * from a compatible version: every field and the checksum are verified, so arbitrary
 * bytes produce an error, never a crash. Indexes saved before an encoding change are
 * refused rather than loaded, since their codes would decode to the wrong vectors. */
size_t    zq_index_serialized_size(const zq_index *index);
zq_status zq_index_serialize(const zq_index *index, void *buf, size_t len, size_t *written);
zq_status zq_index_deserialize(const void *buf, size_t len, zq_index **out);

size_t   zq_index_count(const zq_index *index);
size_t   zq_index_bytes_per_vector(const zq_index *index);
uint32_t zq_index_dim(const zq_index *index);

/* The configuration the index was created with, or saved with if it was loaded, for
 * when all you have is a file. Passing it to zq_index_create gives an index that
 * encodes and scores identically, except for calibration, which that does not carry:
 * zq_index_calibrated is non-zero if the index was calibrated. An index saved through
 * the Zig API with options zq_config cannot express gives ZQ_ERR_UNSUPPORTED. */
zq_status zq_index_config(const zq_index *index, zq_config *out);
int       zq_index_calibrated(const zq_index *index);

/* threads <= 1 searches on the calling thread; greater spreads queries across workers,
 * up to ZQ_MAX_THREADS. Capacity is batch, or threads*batch when threaded. Sizes whose
 * product overflows are rejected with ZQ_ERR_INVALID. */
zq_status zq_searcher_create(zq_index *index, size_t batch, size_t k,
                             size_t threads, zq_searcher **out);
void      zq_searcher_free(zq_searcher *searcher);
size_t    zq_searcher_capacity(const zq_searcher *searcher);

/* Writes nq*k ids and scores, query-major. Caller owns both buffers. The searcher must
 * have been created from this index; one from any other index is refused with
 * ZQ_ERR_INVALID.
 * Where the index holds fewer than k vectors the tail is UINT32_MAX / -inf. */
zq_status zq_search(zq_index *index, zq_searcher *searcher,
                    const float *queries, size_t nq,
                    uint32_t *out_ids, float *out_scores);

/* Search, then rescore each query's candidates exactly against the original vectors
 * and keep the best k (at most the searcher's k, which is how many candidates are
 * fetched: over-fetching 4-10x and reranking recovers near-exact top-k). `originals`
 * holds n_originals row-major vectors of dim floats, row i being the vector added as
 * id i; it must cover every id in the index. Writes nq * k ids and exact scores, padded
 * as zq_search pads. Scores use the index's metric. */
zq_status zq_search_rerank(zq_index *index, zq_searcher *searcher,
                           const float *queries, size_t nq,
                           const float *originals, size_t n_originals, size_t k,
                           uint32_t *ids, float *scores);

/* Score specific stored vectors against one query: scores[j] is the estimate search
 * would report for ids[j]. For ranking a candidate set chosen elsewhere (a metadata
 * filter, another index) without scanning the whole corpus. An id at or past
 * zq_index_count gives ZQ_ERR_INVALID. */
zq_status zq_score_ids(zq_index *index, zq_searcher *searcher, const float *query,
                       const uint32_t *ids, size_t n, float *scores);

/* ── Codec: encode and decode without an index ──────────────────────────────────
 *
 * The index answers "which stored vectors best match this query". A KV cache asks
 * something else: store these vectors compactly and give them back. Attention needs
 * every score rather than the top few, and needs them accurate in absolute terms, so
 * the index's estimator - whose per-vector correction is fitted to preserve ranking -
 * is the wrong tool there and measured considerably worse than reconstruction.
 *
 * Codes are bit-packed, so zq_codec_code_bytes is the storage actually consumed.
 *
 * A codec owns its scratch space and is not safe for concurrent use; create one per
 * thread, as with zq_searcher.
 */

typedef struct zq_codec zq_codec;

typedef struct {
    /* 1..ZQ_MAX_DIM. */
    uint32_t dim;
    /* 1..6, the codebook width directly: unlike the index there is no residual
     * sketch reserving a bit. */
    uint8_t  bits;
    uint64_t seed;
} zq_codec_config;

zq_status zq_codec_create(const zq_codec_config *config, zq_codec **out);
void      zq_codec_free(zq_codec *codec);

/* Packed bytes of code per vector. Norms are stored separately, one float each, and
 * both are required to decode. */
size_t   zq_codec_code_bytes(const zq_codec *codec);
uint32_t zq_codec_dim(const zq_codec *codec);

/* Encode n row-major vectors: codes holds n*zq_codec_code_bytes(), norms holds n. */
zq_status zq_codec_encode(zq_codec *codec, const float *rows, size_t n,
                          uint8_t *codes, float *norms);

/* Decode n vectors into rows, which holds n*dim floats. */
zq_status zq_codec_decode(zq_codec *codec, const uint8_t *codes, const float *norms,
                          size_t n, float *rows);

#ifdef __cplusplus
}
#endif
#endif /* ZQUANT_H */
