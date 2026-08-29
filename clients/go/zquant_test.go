package zquant

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
)

func corpus(n, d int, seed int64, offset float32) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n*d)
	for i := 0; i < n; i++ {
		var sq float64
		for j := 0; j < d; j++ {
			v := float32(r.NormFloat64()) + offset
			out[i*d+j] = v
			sq += float64(v) * float64(v)
		}
		inv := float32(1 / math.Sqrt(sq))
		for j := 0; j < d; j++ {
			out[i*d+j] *= inv
		}
	}
	return out
}

func exactTopK(x, q []float32, d, k int) [][]int {
	n := len(x) / d
	out := make([][]int, len(q)/d)
	for i := range out {
		type sc struct {
			s float32
			j int
		}
		all := make([]sc, n)
		for j := 0; j < n; j++ {
			var s float32
			for c := 0; c < d; c++ {
				s += q[i*d+c] * x[j*d+c]
			}
			all[j] = sc{s, j}
		}
		sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
		ids := make([]int, k)
		for t := 0; t < k; t++ {
			ids[t] = all[t].j
		}
		out[i] = ids
	}
	return out
}

func TestVersion(t *testing.T) {
	if Version() == "" {
		t.Fatal("empty version")
	}
}

func TestSelfRetrieval(t *testing.T) {
	const d, n = 64, 2000
	x := corpus(n, d, 1, 0)
	ix, err := New(Config{Dim: d, Bits: 5})
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	if err := ix.Add(x); err != nil {
		t.Fatal(err)
	}
	if ix.Len() != n {
		t.Fatalf("Len = %d, want %d", ix.Len(), n)
	}
	res, err := ix.Search(x[:50*d], 10, 1)
	if err != nil {
		t.Fatal(err)
	}
	self := 0
	for i := 0; i < 50; i++ {
		ids, _ := res.At(i)
		for _, id := range ids {
			if int(id) == i {
				self++
				break
			}
		}
	}
	if self < 48 {
		t.Fatalf("self-retrieval %d/50", self)
	}
}

func TestRecallAgainstExact(t *testing.T) {
	const d, n, nq = 128, 3000, 100
	x, q := corpus(n, d, 2, 0), corpus(nq, d, 3, 0)
	want := exactTopK(x, q, d, 10)
	ix, _ := New(Config{Dim: d, Bits: 5})
	defer ix.Close()
	if err := ix.Add(x); err != nil {
		t.Fatal(err)
	}
	res, err := ix.Search(q, 10, 1)
	if err != nil {
		t.Fatal(err)
	}
	hits := 0
	for i := range want {
		got, _ := res.At(i)
		set := map[int]bool{}
		for _, id := range got {
			set[int(id)] = true
		}
		for _, j := range want[i] {
			if set[j] {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(len(want)*10)
	if recall < 0.85 {
		t.Fatalf("recall %.3f too low", recall)
	}
}

func TestCalibrationHelpsOnOffsetData(t *testing.T) {
	// The centroid rule as a test: calibration pays when the corpus sits off-origin.
	const d, n, nq = 128, 4000, 100
	x, q := corpus(n, d, 4, 0.85), corpus(nq, d, 5, 0.85)
	want := exactTopK(x, q, d, 10)

	recall := func(calibrate bool) float64 {
		ix, _ := New(Config{Dim: d, Bits: 3})
		defer ix.Close()
		if calibrate {
			if err := ix.Calibrate(x[:2000*d]); err != nil {
				t.Fatal(err)
			}
		}
		if err := ix.Add(x); err != nil {
			t.Fatal(err)
		}
		res, err := ix.Search(q, 10, 1)
		if err != nil {
			t.Fatal(err)
		}
		hits := 0
		for i := range want {
			got, _ := res.At(i)
			set := map[int]bool{}
			for _, id := range got {
				set[int(id)] = true
			}
			for _, j := range want[i] {
				if set[j] {
					hits++
				}
			}
		}
		return float64(hits) / float64(len(want)*10)
	}
	on, off := recall(true), recall(false)
	if on <= off+0.03 {
		t.Fatalf("calibration did not help: %.3f against %.3f", on, off)
	}
}

func TestThreadedAgreesWithSingle(t *testing.T) {
	const d = 64
	ix, _ := New(Config{Dim: d, Bits: 5})
	defer ix.Close()
	ix.Add(corpus(3000, d, 6, 0))
	q := corpus(64, d, 7, 0)
	a, err := ix.Search(q, 10, 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := ix.Search(q, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.IDs {
		if a.IDs[i] != b.IDs[i] {
			t.Fatalf("threaded differs at %d: %d against %d", i, a.IDs[i], b.IDs[i])
		}
	}
}

func TestChunksBeyondSearcherCapacity(t *testing.T) {
	const d = 32
	ix, _ := New(Config{Dim: d, Bits: 5})
	defer ix.Close()
	ix.Add(corpus(1000, d, 8, 0))
	q := corpus(300, d, 9, 0)
	many, err := ix.Search(q, 5, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 300; i += 41 {
		one, err := ix.Search(q[i*d:(i+1)*d], 5, 1)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := many.At(i)
		for j := range one.IDs {
			if one.IDs[j] != got[j] {
				t.Fatalf("query %d differs at %d", i, j)
			}
		}
	}
}

func TestFewerVectorsThanK(t *testing.T) {
	const d = 16
	ix, _ := New(Config{Dim: d, Bits: 5})
	defer ix.Close()
	x := corpus(3, d, 10, 0)
	ix.Add(x)
	res, err := ix.Search(x[:d], 10, 1)
	if err != nil {
		t.Fatal(err)
	}
	ids, scores := res.At(0)
	for j := 3; j < 10; j++ {
		if ids[j] != math.MaxUint32 {
			t.Fatalf("id %d = %d, want MaxUint32", j, ids[j])
		}
		if !math.IsInf(float64(scores[j]), -1) {
			t.Fatalf("score %d = %v, want -Inf", j, scores[j])
		}
	}
}

func TestErrors(t *testing.T) {
	const d = 16
	ix, _ := New(Config{Dim: d, Bits: 5})
	defer ix.Close()
	x := corpus(100, d, 11, 0)

	if _, err := ix.Search(x[:d], 10, 1); err == nil {
		t.Fatal("search on an empty index should fail")
	}
	ix.Add(x)
	if err := ix.Calibrate(x); err == nil {
		t.Fatal("calibrate after add should fail")
	}
	if err := ix.Add(make([]float32, d+1)); err == nil {
		t.Fatal("ragged input should fail")
	}
	if _, err := ix.Search(x[:d], 0, 1); err == nil {
		t.Fatal("k=0 should fail")
	}
	if _, err := New(Config{Dim: 0}); err == nil {
		t.Fatal("dim=0 should fail")
	}
	if _, err := New(Config{Dim: 8, Bits: 99}); err == nil {
		t.Fatal("bits=99 should fail")
	}
}

func TestConcurrentSearchers(t *testing.T) {
	// The documented concurrency contract: one Searcher per goroutine.
	const d = 64
	ix, _ := New(Config{Dim: d, Bits: 5})
	defer ix.Close()
	ix.Add(corpus(2000, d, 12, 0))
	q := corpus(32, d, 13, 0)

	want, err := ix.Search(q, 10, 1)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, err := ix.NewSearcher(10, 1)
			if err != nil {
				errs <- err
				return
			}
			defer s.Close()
			for i := 0; i < 20; i++ {
				got, err := ix.SearchWith(s, q)
				if err != nil {
					errs <- err
					return
				}
				for j := range want.IDs {
					if got.IDs[j] != want.IDs[j] {
						errs <- err
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent search: %v", err)
	}
}
