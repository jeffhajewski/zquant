package zquant

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These tests are aimed at -race and at the allocator: each would have touched a freed
// or concurrently mutated native handle before the binding held a lock across every
// native call and kept its handles alive until the call returned.

func TestCloseIsIdempotentAndLaterCallsFail(t *testing.T) {
	const d = 16
	ix, err := New(Config{Dim: d})
	if err != nil {
		t.Fatal(err)
	}
	x := corpus(50, d, 30, 0)
	if err := ix.Add(x); err != nil {
		t.Fatal(err)
	}
	s, err := ix.NewSearcher(5, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ix.Search(x[:d], 5, 1); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		if err := ix.Close(); err != nil {
			t.Fatalf("Close #%d: %v", i, err)
		}
	}

	checks := map[string]error{
		"Add":       ix.Add(x),
		"Calibrate": ix.Calibrate(x),
	}
	_, checks["Search"] = ix.Search(x[:d], 5, 1)
	_, checks["SearchWith"] = ix.SearchWith(s, x[:d])
	_, checks["NewSearcher"] = ix.NewSearcher(5, 1)
	for name, err := range checks {
		if !errors.Is(err, ErrClosed) {
			t.Errorf("%s after Close: got %v, want ErrClosed", name, err)
		}
	}
	if ix.Len() != 0 || ix.BytesPerVector() != 0 {
		t.Errorf("closed index reports Len %d, BytesPerVector %d", ix.Len(), ix.BytesPerVector())
	}

	// A searcher outliving its index can still be released, any number of times.
	for i := 0; i < 3; i++ {
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestClosedSearcherFails(t *testing.T) {
	const d = 16
	ix, _ := New(Config{Dim: d})
	defer ix.Close()
	x := corpus(50, d, 31, 0)
	ix.Add(x)
	s, _ := ix.NewSearcher(5, 1)
	s.Close()
	if _, err := ix.SearchWith(s, x[:d]); !errors.Is(err, ErrClosed) {
		t.Fatalf("search with a closed searcher: got %v, want ErrClosed", err)
	}
}

func TestSearcherFromAnotherIndexIsRefused(t *testing.T) {
	const d = 16
	a, _ := New(Config{Dim: d})
	defer a.Close()
	b, _ := New(Config{Dim: d})
	defer b.Close()
	x := corpus(50, d, 32, 0)
	a.Add(x)
	b.Add(x)
	s, _ := a.NewSearcher(5, 1)
	defer s.Close()
	if _, err := b.SearchWith(s, x[:d]); err == nil {
		t.Fatal("a searcher from another index should be refused")
	}
}

// Distinct searchers on one index search in parallel. Each result is compared against
// a single-threaded reference, so a data race that corrupted scratch would show as a
// mismatch even where the race detector is not running.
func TestConcurrentDistinctSearchers(t *testing.T) {
	const d, goroutines, rounds = 64, 8, 25
	ix, _ := New(Config{Dim: d, Bits: 5})
	defer ix.Close()
	if err := ix.Add(corpus(2000, d, 33, 0)); err != nil {
		t.Fatal(err)
	}
	q := corpus(40, d, 34, 0)
	want, err := ix.Search(q, 10, 1)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			// Mix single- and multi-threaded searchers; both must agree with the reference.
			s, err := ix.NewSearcher(10, 1+g%3)
			if err != nil {
				errs <- err
				return
			}
			defer s.Close()
			for i := 0; i < rounds; i++ {
				got, err := ix.SearchWith(s, q)
				if err != nil {
					errs <- err
					return
				}
				for j := range want.IDs {
					if got.IDs[j] != want.IDs[j] || got.Scores[j] != want.Scores[j] {
						errs <- fmt.Errorf("goroutine %d round %d: result %d differs", g, i, j)
						return
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// Sharing one Searcher between goroutines breaks the native contract; the binding
// serializes it rather than letting two searches scribble over the same scratch.
func TestSharedSearcherIsSerialized(t *testing.T) {
	const d = 32
	ix, _ := New(Config{Dim: d})
	defer ix.Close()
	ix.Add(corpus(1000, d, 35, 0))
	q := corpus(16, d, 36, 0)
	s, _ := ix.NewSearcher(10, 1)
	defer s.Close()
	want, _ := ix.SearchWith(s, q)

	var wg sync.WaitGroup
	var bad atomic.Int32
	for g := 0; g < 6; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				got, err := ix.SearchWith(s, q)
				if err != nil {
					bad.Add(1)
					return
				}
				for j := range want.IDs {
					if got.IDs[j] != want.IDs[j] {
						bad.Add(1)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	if bad.Load() != 0 {
		t.Fatalf("%d goroutines saw an error or a wrong result", bad.Load())
	}
}

// Close lands while searches are running. Every search must either finish normally or
// report ErrClosed; none may use the handle after it is freed.
func TestCloseDuringSearches(t *testing.T) {
	const d = 64
	for trial := 0; trial < 5; trial++ {
		ix, _ := New(Config{Dim: d})
		ix.Add(corpus(3000, d, 37, 0))
		q := corpus(64, d, 38, 0)

		var wg sync.WaitGroup
		var started sync.WaitGroup
		errs := make(chan error, 16)
		for g := 0; g < 8; g++ {
			wg.Add(1)
			started.Add(1)
			go func(g int) {
				defer wg.Done()
				s, err := ix.NewSearcher(10, 1+g%2)
				if err != nil {
					started.Done()
					errs <- err
					return
				}
				defer s.Close()
				started.Done()
				for {
					var err error
					if g%2 == 0 {
						_, err = ix.SearchWith(s, q)
					} else {
						_, err = ix.Search(q, 10, 2)
					}
					if errors.Is(err, ErrClosed) {
						return
					}
					if err != nil {
						errs <- err
						return
					}
				}
			}(g)
		}
		started.Wait()
		time.Sleep(time.Millisecond)
		// Several closers at once: Close must be idempotent under contention too.
		var closers sync.WaitGroup
		for c := 0; c < 3; c++ {
			closers.Add(1)
			go func() { defer closers.Done(); ix.Close() }()
		}
		closers.Wait()
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Fatalf("trial %d: %v", trial, err)
		}
	}
}

// Add runs while searches are in flight. The native library forbids the overlap, so
// the binding must hold searches off for the duration of each Add; every search sees
// some whole prefix of the corpus, never a half-appended batch.
func TestAddDuringSearches(t *testing.T) {
	const d, batch, batches = 32, 200, 20
	ix, _ := New(Config{Dim: d})
	defer ix.Close()
	x := corpus(batch*batches, d, 39, 0)
	if err := ix.Add(x[:batch*d]); err != nil {
		t.Fatal(err)
	}
	q := corpus(8, d, 40, 0)

	stop := make(chan struct{})
	var wg, started sync.WaitGroup
	errs := make(chan error, 8)
	for g := 0; g < 4; g++ {
		wg.Add(1)
		started.Add(1)
		go func(g int) {
			defer wg.Done()
			first := true
			defer func() {
				if first {
					started.Done()
				}
			}()
			s, err := ix.NewSearcher(10, 1)
			if err != nil {
				errs <- err
				return
			}
			defer s.Close()
			for {
				select {
				case <-stop:
					return
				default:
				}
				var res Results
				if g%2 == 0 {
					res, err = ix.SearchWith(s, q)
				} else {
					res, err = ix.Search(q, 10, 1)
				}
				if err != nil {
					errs <- err
					return
				}
				// Ids past what any Add has completed would mean a search saw a torn add.
				n := uint32(ix.Len())
				for _, id := range res.IDs {
					if id >= n {
						errs <- fmt.Errorf("id %d from an index of %d", id, n)
						return
					}
				}
				if first {
					first = false
					started.Done()
				}
			}
		}(g)
	}
	// Start adding only once every searcher is running, so the two really overlap.
	started.Wait()
	for b := 1; b < batches; b++ {
		if err := ix.Add(x[b*batch*d : (b+1)*batch*d]); err != nil {
			t.Fatal(err)
		}
		runtime.Gosched()
	}
	close(stop)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if ix.Len() != batch*batches {
		t.Fatalf("Len = %d, want %d", ix.Len(), batch*batches)
	}
}

// Indexes and searchers dropped without Close are released by finalizers, which run on
// the collector's goroutine while other goroutines are still making native calls. A
// collector running continuously maximises the chance of finalizing an object whose
// handle a native call is still using, which runtime.KeepAlive exists to prevent.
func TestFinalizersUnderConstantGC(t *testing.T) {
	const d = 32
	x := corpus(300, d, 41, 0)
	q := corpus(4, d, 42, 0)

	stop := make(chan struct{})
	var gcDone sync.WaitGroup
	gcDone.Add(1)
	go func() {
		defer gcDone.Done()
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				// Nothing here is closed: everything is left to the finalizers.
				ix, err := New(Config{Dim: d})
				if err != nil {
					errs <- err
					return
				}
				if err := ix.Add(x); err != nil {
					errs <- err
					return
				}
				s, err := ix.NewSearcher(5, 1)
				if err != nil {
					errs <- err
					return
				}
				if _, err := ix.SearchWith(s, q); err != nil {
					errs <- err
					return
				}
				// The index's last use is inside this call.
				if _, err := ix.Search(q, 5, 1); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	gcDone.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	// Give the finalizers queued by the last collections a chance to run, so a crash
	// in one happens inside this test rather than later.
	for i := 0; i < 3; i++ {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
}
