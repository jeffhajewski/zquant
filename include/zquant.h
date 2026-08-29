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

#define ZQ_METRIC_INNER_PRODUCT 0
#define ZQ_METRIC_COSINE        1
#define ZQ_METRIC_L2            2

typedef struct zq_index zq_index;
typedef struct zq_searcher zq_searcher;

typedef struct {
    uint32_t dim;
    /* Total bit budget per coordinate, 2..6. The scalar codebook uses bits-1. */
    uint8_t  bits;
    int      metric;
    uint64_t seed;
    /* Non-zero keeps codes packed (smaller); zero dequantizes to int8 (faster, ~2x memory). */
    int      compact;
} zq_config;

const char *zq_version(void);
const char *zq_status_string(zq_status status);

zq_status zq_index_create(const zq_config *config, zq_index **out);
void      zq_index_free(zq_index *index);

/* Fit per-coordinate shift and scale from a sample. Must precede any add.
 *
 * Worth doing when the corpus centroid sits away from the origin: compute
 * ||mean(x/||x||)|| on a sample and expect a gain above roughly 0.3, and none below.
 * It is not free - on low-rank zero-mean data it can cost recall. */
zq_status zq_index_calibrate(zq_index *index, const float *rows, size_t n);

/* Append n row-major vectors of dim floats each. */
zq_status zq_index_add(zq_index *index, const float *rows, size_t n);

size_t   zq_index_count(const zq_index *index);
size_t   zq_index_bytes_per_vector(const zq_index *index);
uint32_t zq_index_dim(const zq_index *index);

/* threads <= 1 searches on the calling thread; greater spreads queries across workers.
 * Capacity is batch, or threads*batch when threaded. */
zq_status zq_searcher_create(zq_index *index, size_t batch, size_t k,
                             size_t threads, zq_searcher **out);
void      zq_searcher_free(zq_searcher *searcher);
size_t    zq_searcher_capacity(const zq_searcher *searcher);

/* Writes nq*k ids and scores, query-major. Caller owns both buffers.
 * Where the index holds fewer than k vectors the tail is UINT32_MAX / -inf. */
zq_status zq_search(zq_index *index, zq_searcher *searcher,
                    const float *queries, size_t nq,
                    uint32_t *out_ids, float *out_scores);

#ifdef __cplusplus
}
#endif
#endif /* ZQUANT_H */
