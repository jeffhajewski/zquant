package zquant

/*
#include "zquant.h"
*/
import "C"

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

// Searcher holds the scratch space a search needs. It belongs to the index that made it
// and works only with that index.
//
// A Searcher runs one search at a time: calls that share one wait for each other. To
// search concurrently, give each goroutine its own.
type Searcher struct {
	// mu guards handle and is held for the whole of each native call using it, so Close
	// cannot free the scratch under a search and two searches cannot share it.
	mu       sync.Mutex
	handle   *C.zq_searcher
	owner    uint64 // Index.id of the index that made it
	k        int
	capacity int
}

// errSearcherClosed is what every call on a closed searcher returns.
const errSearcherClosed = closedError("searcher")

// NewSearcher creates search scratch for a given k and thread count. threads greater
// than one splits a batch across workers, which only pays for batches at least that
// large.
func (ix *Index) NewSearcher(k, threads int) (*Searcher, error) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.newSearcherLocked(k, threads)
}

// newSearcherLocked needs ix.mu held, shared or exclusive.
func (ix *Index) newSearcherLocked(k, threads int) (*Searcher, error) {
	if ix.handle == nil {
		return nil, errIndexClosed
	}
	if k < 1 {
		return nil, errors.New("zquant: k must be at least 1")
	}
	if threads < 1 {
		threads = 1
	}
	var handle *C.zq_searcher
	const batch = 32
	st := C.zq_searcher_create(ix.handle, batch, C.size_t(k), C.size_t(threads), &handle)
	runtime.KeepAlive(ix)
	if err := check(st, "searcher"); err != nil {
		return nil, err
	}
	s := &Searcher{handle: handle, owner: ix.id, k: k, capacity: int(C.zq_searcher_capacity(handle))}
	// As for Index: a backstop for a Searcher dropped without Close. It holds no
	// pointer to its index, so the two never form a cycle the collector cannot order.
	runtime.SetFinalizer(s, (*Searcher).Close)
	return s, nil
}

// K reports how many results per query the searcher returns.
func (s *Searcher) K() int { return s.k }

// Close releases the searcher, waiting for a search in flight on it to finish. It is
// idempotent.
func (s *Searcher) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle != nil {
		C.zq_searcher_free(s.handle)
		s.handle = nil
		runtime.SetFinalizer(s, nil)
	}
	return nil
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

// acquire locks s for a call on ix, which must already hold ix.mu, and checks that both
// are open and that s belongs to ix. On success the caller must unlock s.mu.
func (ix *Index) acquire(s *Searcher) error {
	if ix.handle == nil {
		return errIndexClosed
	}
	s.mu.Lock()
	if s.handle == nil {
		s.mu.Unlock()
		return errSearcherClosed
	}
	if s.owner != ix.id {
		s.mu.Unlock()
		return &Error{Status: C.ZQ_ERR_INVALID, Context: "searcher belongs to a different index"}
	}
	return nil
}

// SearchWith runs a search using a caller-owned Searcher, which is the form to use when
// searching concurrently: give each goroutine its own.
func (ix *Index) SearchWith(s *Searcher, queries []float32) (Results, error) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.searchLocked(s, queries)
}

// searchLocked needs ix.mu held shared.
func (ix *Index) searchLocked(s *Searcher, queries []float32) (Results, error) {
	nq, err := ix.rows(queries, "queries")
	if err != nil {
		return Results{}, err
	}
	if err := ix.acquire(s); err != nil {
		return Results{}, err
	}
	defer s.mu.Unlock()
	if ix.lenLocked() == 0 {
		return Results{}, &Error{Status: C.ZQ_ERR_STATE, Context: "search on an empty index"}
	}

	ids := make([]uint32, nq*s.k)
	scores := make([]float32, nq*s.k)
	// The native call takes at most capacity queries; chunk here rather than making
	// that the caller's problem.
	for start := 0; start < nq; start += s.capacity {
		n := min(s.capacity, nq-start)
		st := C.zq_search(
			ix.handle, s.handle,
			(*C.float)(&queries[start*ix.dim]), C.size_t(n),
			(*C.uint32_t)(unsafe.Pointer(&ids[start*s.k])),
			(*C.float)(&scores[start*s.k]),
		)
		runtime.KeepAlive(ix)
		runtime.KeepAlive(s)
		if err := check(st, "search"); err != nil {
			return Results{}, err
		}
	}
	return Results{IDs: ids, Scores: scores, K: s.k}, nil
}

// Search is the convenience form: it reuses a Searcher cached per (k, threads).
//
// Safe to call concurrently. Calls with the same k and threads share one cached
// searcher and so take turns; for concurrent throughput use NewSearcher and SearchWith
// with one searcher per goroutine.
func (ix *Index) Search(queries []float32, k, threads int) (Results, error) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if ix.handle == nil {
		return Results{}, errIndexClosed
	}
	key := [2]int{k, threads}
	ix.cacheMu.Lock()
	s, ok := ix.cache[key]
	if !ok {
		var err error
		if s, err = ix.newSearcherLocked(k, threads); err != nil {
			ix.cacheMu.Unlock()
			return Results{}, err
		}
		ix.cache[key] = s
	}
	ix.cacheMu.Unlock()
	return ix.searchLocked(s, queries)
}
