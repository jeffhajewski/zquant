package zquant

/*
#include "zquant.h"
*/
import "C"

import (
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
		return nil, invalid("k %d must be at least 1", k)
	}
	if threads > MaxThreads {
		return nil, invalid("threads %d exceeds %d", threads, MaxThreads)
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

// SetProbes sets how many lists each query scans on a partitioned index, from then on.
// Values above the number of lists scan them all. More probes find more of the true
// neighbours and cost proportionally more time; a new searcher starts at an eighth of
// the lists. On a searcher of an index that is not partitioned it returns an error
// matching ErrState.
func (s *Searcher) SetProbes(probes int) error {
	if probes < 1 {
		return invalid("probes %d must be at least 1", probes)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return errSearcherClosed
	}
	st := C.zq_searcher_set_probes(s.handle, C.size_t(probes))
	runtime.KeepAlive(s)
	return check(st, "set probes")
}

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

// SearchRerank searches, then rescores each query's candidates exactly against the
// original vectors and returns the best k of them, with exact scores under the index's
// metric.
//
// The searcher's K is how many candidates are fetched, and k must be at most that.
// Fetching 4-10x more candidates than you keep, and reranking, recovers close to the
// exact top k. originals holds the vectors as added, row-major, row i being id i, and
// must cover every vector in the index.
func (ix *Index) SearchRerank(s *Searcher, queries, originals []float32, k int) (Results, error) {
	nq, err := ix.rows(queries, "queries")
	if err != nil {
		return Results{}, err
	}
	nOrig, err := ix.rows(originals, "originals")
	if err != nil {
		return Results{}, err
	}
	if k < 1 || k > s.k {
		return Results{}, invalid("rerank k %d is outside 1..%d, the searcher's candidate count", k, s.k)
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if err := ix.acquire(s); err != nil {
		return Results{}, err
	}
	defer s.mu.Unlock()
	count := ix.lenLocked()
	if count == 0 {
		return Results{}, &Error{Status: C.ZQ_ERR_STATE, Context: "search on an empty index"}
	}
	if nOrig < count {
		return Results{}, invalid("originals hold %d vectors but the index holds %d", nOrig, count)
	}

	ids := make([]uint32, nq*k)
	scores := make([]float32, nq*k)
	for start := 0; start < nq; start += s.capacity {
		n := min(s.capacity, nq-start)
		st := C.zq_search_rerank(
			ix.handle, s.handle,
			(*C.float)(&queries[start*ix.dim]), C.size_t(n),
			(*C.float)(&originals[0]), C.size_t(nOrig), C.size_t(k),
			(*C.uint32_t)(unsafe.Pointer(&ids[start*k])),
			(*C.float)(&scores[start*k]),
		)
		runtime.KeepAlive(ix)
		runtime.KeepAlive(s)
		if err := check(st, "search rerank"); err != nil {
			return Results{}, err
		}
	}
	return Results{IDs: ids, Scores: scores, K: k}, nil
}

// ScoreIDs scores chosen stored vectors against one query: the result's j-th score is
// the estimate a search would report for ids[j]. It is for ranking a candidate set
// picked some other way, by a metadata filter or another index, without scanning the
// whole corpus. On a partitioned index every id is scored, whatever list it is in. An
// id at or past Len is an error matching ErrInvalid.
func (ix *Index) ScoreIDs(s *Searcher, query []float32, ids []uint32) ([]float32, error) {
	if len(query) != ix.dim {
		return nil, invalid("query length %d is not dim %d", len(query), ix.dim)
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if err := ix.acquire(s); err != nil {
		return nil, err
	}
	defer s.mu.Unlock()
	scores := make([]float32, len(ids))
	if len(ids) == 0 {
		return scores, nil
	}
	st := C.zq_score_ids(
		ix.handle, s.handle, (*C.float)(&query[0]),
		(*C.uint32_t)(unsafe.Pointer(&ids[0])), C.size_t(len(ids)),
		(*C.float)(&scores[0]),
	)
	runtime.KeepAlive(ix)
	runtime.KeepAlive(s)
	if err := check(st, "score ids"); err != nil {
		return nil, err
	}
	return scores, nil
}
