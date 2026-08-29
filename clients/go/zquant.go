// Package zquant provides a flat quantized vector index backed by TurboQuant.
//
//	index, err := zquant.New(zquant.Config{Dim: 256, Bits: 5})
//	defer index.Close()
//	index.Add(vectors)                        // len(vectors) == n*Dim
//	res, err := index.Search(queries, 10, 8)  // k=10, 8 threads
//
// Vectors are float32, row-major, one per row.
package zquant

/*
#include <stdlib.h>
#include "zquant.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Metric selects how a stored vector is scored against a query.
type Metric int

const (
	InnerProduct Metric = C.ZQ_METRIC_INNER_PRODUCT
	Cosine       Metric = C.ZQ_METRIC_COSINE
	L2           Metric = C.ZQ_METRIC_L2
)

// Error is a failure reported by the native library, carrying its status code.
type Error struct {
	Status  int
	Context string
}

func (e *Error) Error() string {
	return fmt.Sprintf("zquant: %s: %s (%d)", e.Context, C.GoString(C.zq_status_string(C.int(e.Status))), e.Status)
}

func check(status C.int, context string) error {
	if status == C.ZQ_OK {
		return nil
	}
	return &Error{Status: int(status), Context: context}
}

// Version reports the native library version.
func Version() string { return C.GoString(C.zq_version()) }

// Config describes an index. The zero value is not usable; Dim is required.
type Config struct {
	Dim int
	// Bits is the total budget per coordinate, 2..6. The scalar codebook uses Bits-1.
	// Zero means 5, which is a good default: 132 B/vector at R@10 0.916 on 256-d
	// embeddings.
	Bits   int
	Metric Metric
	Seed   uint64
	// Expanded stores dequantized int8 rather than packed codes: roughly twice the
	// memory for some throughput.
	Expanded bool
}

// Index is a flat (exhaustive) quantized index. Every vector is scanned for every
// query, so recall is bounded only by quantization and cost grows linearly with the
// corpus.
//
// An Index is safe for concurrent Search calls. Add and Calibrate are not, and must not
// run concurrently with anything else on the same Index.
type Index struct {
	handle *C.zq_index
	dim    int

	mu        sync.Mutex
	searchers map[[2]int]*Searcher
}

// New creates an empty index.
func New(cfg Config) (*Index, error) {
	if cfg.Dim <= 0 {
		return nil, errors.New("zquant: Dim must be positive")
	}
	bits := cfg.Bits
	if bits == 0 {
		bits = 5
	}
	seed := cfg.Seed
	if seed == 0 {
		seed = 0x5EED
	}
	compact := C.int(1)
	if cfg.Expanded {
		compact = 0
	}
	c := C.zq_config{
		dim:     C.uint32_t(cfg.Dim),
		bits:    C.uint8_t(bits),
		metric:  C.int(cfg.Metric),
		seed:    C.uint64_t(seed),
		compact: compact,
	}
	var handle *C.zq_index
	if err := check(C.zq_index_create(&c, &handle), "create"); err != nil {
		return nil, err
	}
	ix := &Index{handle: handle, dim: cfg.Dim, searchers: map[[2]int]*Searcher{}}
	// A finalizer is a backstop, not the contract: callers should Close. Without it a
	// dropped Index leaks native memory the garbage collector cannot see.
	runtime.SetFinalizer(ix, func(i *Index) { i.Close() })
	return ix, nil
}

// Close releases the index and every searcher created from it. It is idempotent.
func (ix *Index) Close() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	for _, s := range ix.searchers {
		s.close()
	}
	ix.searchers = map[[2]int]*Searcher{}
	if ix.handle != nil {
		C.zq_index_free(ix.handle)
		ix.handle = nil
		runtime.SetFinalizer(ix, nil)
	}
	return nil
}

// Dim reports the vector dimension.
func (ix *Index) Dim() int { return ix.dim }

// Len reports how many vectors the index holds.
func (ix *Index) Len() int { return int(C.zq_index_count(ix.handle)) }

// BytesPerVector reports storage per vector, codes and scalars together.
func (ix *Index) BytesPerVector() int { return int(C.zq_index_bytes_per_vector(ix.handle)) }

func (ix *Index) rows(v []float32, name string) (int, error) {
	if ix.handle == nil {
		return 0, errors.New("zquant: index is closed")
	}
	if len(v) == 0 || len(v)%ix.dim != 0 {
		return 0, fmt.Errorf("zquant: %s length %d is not a positive multiple of dim %d", name, len(v), ix.dim)
	}
	return len(v) / ix.dim, nil
}

// Calibrate fits a per-coordinate shift and scale from a sample. It must precede Add.
//
// Worth doing when the corpus centroid sits away from the origin — the norm of the mean
// of the unit-normalized vectors. Above about 0.3 expect a real gain; below it expect
// none. It is not free: on low-rank zero-mean data it has cost recall.
func (ix *Index) Calibrate(sample []float32) error {
	n, err := ix.rows(sample, "sample")
	if err != nil {
		return err
	}
	return check(C.zq_index_calibrate(ix.handle, (*C.float)(&sample[0]), C.size_t(n)), "calibrate")
}

// Add appends vectors to the index.
func (ix *Index) Add(vectors []float32) error {
	n, err := ix.rows(vectors, "vectors")
	if err != nil {
		return err
	}
	return check(C.zq_index_add(ix.handle, (*C.float)(&vectors[0]), C.size_t(n)), "add")
}

// Searcher holds the scratch space one search needs. Each goroutine searching
// concurrently must own a distinct Searcher.
type Searcher struct {
	handle   *C.zq_searcher
	k        int
	capacity int
}

// NewSearcher creates search scratch for a given k and thread count. threads greater
// than one splits a batch across workers, which only pays for batches at least that
// large.
func (ix *Index) NewSearcher(k, threads int) (*Searcher, error) {
	if k < 1 {
		return nil, errors.New("zquant: k must be at least 1")
	}
	if threads < 1 {
		threads = 1
	}
	var handle *C.zq_searcher
	const batch = 32
	if err := check(C.zq_searcher_create(ix.handle, batch, C.size_t(k), C.size_t(threads), &handle), "searcher"); err != nil {
		return nil, err
	}
	return &Searcher{handle: handle, k: k, capacity: int(C.zq_searcher_capacity(handle))}, nil
}

// Close releases the searcher. It is idempotent.
func (s *Searcher) Close() error { s.close(); return nil }

func (s *Searcher) close() {
	if s.handle != nil {
		C.zq_searcher_free(s.handle)
		s.handle = nil
	}
}

// Results holds nq*K ids and scores, query-major and best first. Where the index holds
// fewer than K vectors the tail is math.MaxUint32 and negative infinity.
type Results struct {
	IDs    []uint32
	Scores []float32
	K      int
}

// At returns the ids and scores for query i.
func (r Results) At(i int) ([]uint32, []float32) {
	return r.IDs[i*r.K : (i+1)*r.K], r.Scores[i*r.K : (i+1)*r.K]
}

// SearchWith runs a search using a caller-owned Searcher, which is the form to use when
// searching concurrently: give each goroutine its own.
func (ix *Index) SearchWith(s *Searcher, queries []float32) (Results, error) {
	nq, err := ix.rows(queries, "queries")
	if err != nil {
		return Results{}, err
	}
	if s.handle == nil {
		return Results{}, errors.New("zquant: searcher is closed")
	}
	if ix.Len() == 0 {
		return Results{}, &Error{Status: C.ZQ_ERR_STATE, Context: "search on an empty index"}
	}

	ids := make([]uint32, nq*s.k)
	scores := make([]float32, nq*s.k)
	// The native call takes at most capacity queries; chunk here rather than making
	// that the caller's problem.
	for start := 0; start < nq; start += s.capacity {
		n := s.capacity
		if n > nq-start {
			n = nq - start
		}
		st := C.zq_search(
			ix.handle, s.handle,
			(*C.float)(&queries[start*ix.dim]), C.size_t(n),
			(*C.uint32_t)(unsafe.Pointer(&ids[start*s.k])),
			(*C.float)(&scores[start*s.k]),
		)
		if err := check(st, "search"); err != nil {
			return Results{}, err
		}
	}
	return Results{IDs: ids, Scores: scores, K: s.k}, nil
}

// Search is the convenience form: it reuses a cached Searcher per (k, threads).
//
// Safe to call concurrently, but serialized by a mutex, because a Searcher cannot be
// shared across goroutines. For concurrent throughput use NewSearcher and SearchWith
// with one searcher per goroutine.
func (ix *Index) Search(queries []float32, k, threads int) (Results, error) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	key := [2]int{k, threads}
	s, ok := ix.searchers[key]
	if !ok {
		var err error
		if s, err = ix.NewSearcher(k, threads); err != nil {
			return Results{}, err
		}
		ix.searchers[key] = s
	}
	return ix.SearchWith(s, queries)
}
