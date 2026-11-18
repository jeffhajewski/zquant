package zquant

import (
	"errors"
	"math"
	"testing"
)

func TestAddParallelMatchesAdd(t *testing.T) {
	const d, n = 64, 3000
	x, q := corpus(n, d, 60, 0), corpus(40, d, 61, 0)

	ref, _ := New(Config{Dim: d})
	defer ref.Close()
	if err := ref.Add(x); err != nil {
		t.Fatal(err)
	}
	want, err := ref.Search(q, 10, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, threads := range []int{0, 1, 4, 16} {
		ix, _ := New(Config{Dim: d})
		if err := ix.AddParallel(x, threads); err != nil {
			t.Fatalf("threads=%d: %v", threads, err)
		}
		if ix.Len() != n {
			t.Fatalf("threads=%d: Len %d", threads, ix.Len())
		}
		got, err := ix.Search(q, 10, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.IDs {
			if got.IDs[i] != want.IDs[i] || got.Scores[i] != want.Scores[i] {
				t.Fatalf("threads=%d: result %d differs from Add", threads, i)
			}
		}
		ix.Close()
	}

	ix, _ := New(Config{Dim: d})
	defer ix.Close()
	for _, threads := range []int{-1, MaxThreads + 1} {
		if err := ix.AddParallel(x, threads); !errors.Is(err, ErrInvalid) {
			t.Errorf("threads=%d: got %v, want ErrInvalid", threads, err)
		}
	}
}

func TestCalibrateAuto(t *testing.T) {
	const d = 64
	// Off-origin data: the centroid rule says calibrate.
	ix, _ := New(Config{Dim: d, Bits: 3})
	norm, applied, err := ix.CalibrateAuto(corpus(500, d, 62, 0.85))
	if err != nil {
		t.Fatal(err)
	}
	if !applied || norm < 0.3 {
		t.Errorf("offset data: norm %.3f, applied %v; want applied with norm >= 0.3", norm, applied)
	}
	ix.Add(corpus(10, d, 63, 0.85))
	if _, _, err := ix.CalibrateAuto(corpus(10, d, 64, 0.85)); !errors.Is(err, ErrState) {
		t.Errorf("after add: got %v, want ErrState", err)
	}
	ix.Close()

	// Zero-mean data: it should measure and decline.
	ix, _ = New(Config{Dim: d, Bits: 3})
	defer ix.Close()
	norm, applied, err = ix.CalibrateAuto(corpus(500, d, 65, 0))
	if err != nil {
		t.Fatal(err)
	}
	if applied || norm >= 0.3 {
		t.Errorf("zero-mean data: norm %.3f, applied %v; want not applied with norm < 0.3", norm, applied)
	}
	if _, _, err := ix.CalibrateAuto(make([]float32, d+1)); !errors.Is(err, ErrInvalid) {
		t.Errorf("ragged sample: %v", err)
	}
}

func TestSearchRerank(t *testing.T) {
	// 100 queries is more than one searcher's capacity, so this also covers chunking.
	const d, n, nq, k, fetch = 128, 3000, 100, 10, 50
	x, q := corpus(n, d, 66, 0), corpus(nq, d, 67, 0)
	want := exactTopK(x, q, d, k)

	ix, _ := New(Config{Dim: d, Bits: 3})
	defer ix.Close()
	if err := ix.Add(x); err != nil {
		t.Fatal(err)
	}
	s, err := ix.NewSearcher(fetch, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.K() != fetch {
		t.Fatalf("K() = %d", s.K())
	}

	res, err := ix.SearchRerank(s, q, x, k)
	if err != nil {
		t.Fatal(err)
	}
	if res.K != k || len(res.IDs) != nq*k {
		t.Fatalf("got K %d and %d ids", res.K, len(res.IDs))
	}
	hits := 0
	for i := 0; i < nq; i++ {
		ids, scores := res.At(i)
		set := map[int]bool{}
		for j, id := range ids {
			set[int(id)] = true
			// Scores are exact inner products against the originals.
			var dot float64
			for c := 0; c < d; c++ {
				dot += float64(q[i*d+c]) * float64(x[int(id)*d+c])
			}
			if math.Abs(dot-float64(scores[j])) > 1e-4 {
				t.Fatalf("query %d rank %d: score %v, exact %v", i, j, scores[j], dot)
			}
			if j > 0 && scores[j] > scores[j-1] {
				t.Fatalf("query %d: scores not best first", i)
			}
		}
		for _, j := range want[i] {
			if set[j] {
				hits++
			}
		}
	}
	plain, err := ix.Search(q, k, 1)
	if err != nil {
		t.Fatal(err)
	}
	plainHits := 0
	for i := 0; i < nq; i++ {
		ids, _ := plain.At(i)
		set := map[int]bool{}
		for _, id := range ids {
			set[int(id)] = true
		}
		for _, j := range want[i] {
			if set[j] {
				plainHits++
			}
		}
	}
	reranked, base := float64(hits)/float64(nq*k), float64(plainHits)/float64(nq*k)
	t.Logf("recall@%d: %.3f plain, %.3f reranked from %d candidates", k, base, reranked, fetch)
	if reranked < 0.9 || reranked < base+0.05 {
		t.Fatalf("reranking did not recover recall: %.3f plain, %.3f reranked", base, reranked)
	}

	if _, err := ix.SearchRerank(s, q, x, fetch+1); !errors.Is(err, ErrInvalid) {
		t.Errorf("k past the searcher's: got %v, want ErrInvalid", err)
	}
	if _, err := ix.SearchRerank(s, q, x, 0); !errors.Is(err, ErrInvalid) {
		t.Errorf("k=0: got %v, want ErrInvalid", err)
	}
	if _, err := ix.SearchRerank(s, q, x[:(n-1)*d], k); !errors.Is(err, ErrInvalid) {
		t.Errorf("originals short of the index: got %v, want ErrInvalid", err)
	}
}

func TestScoreIDs(t *testing.T) {
	const d, n = 64, 500
	x, q := corpus(n, d, 68, 0), corpus(1, d, 69, 0)
	ix, _ := New(Config{Dim: d})
	defer ix.Close()
	ix.Add(x)
	s, _ := ix.NewSearcher(20, 1)
	defer s.Close()

	res, err := ix.SearchWith(s, q)
	if err != nil {
		t.Fatal(err)
	}
	ids, want := res.At(0)
	// Ask in reverse order, to show scores follow the ids given rather than rank.
	rev := make([]uint32, len(ids))
	for i := range ids {
		rev[i] = ids[len(ids)-1-i]
	}
	got, err := ix.ScoreIDs(s, q, rev)
	if err != nil {
		t.Fatal(err)
	}
	for i := range rev {
		w := want[len(ids)-1-i]
		if math.Abs(float64(got[i]-w)) > 1e-5*math.Max(1, math.Abs(float64(w))) {
			t.Fatalf("id %d: ScoreIDs %v, search %v", rev[i], got[i], w)
		}
	}

	if got, err := ix.ScoreIDs(s, q, nil); err != nil || len(got) != 0 {
		t.Errorf("no ids: %v, %v", got, err)
	}
	if _, err := ix.ScoreIDs(s, q, []uint32{0, n}); !errors.Is(err, ErrInvalid) {
		t.Errorf("id past Len: got %v, want ErrInvalid", err)
	}
	if _, err := ix.ScoreIDs(s, q[:d-1], []uint32{0}); !errors.Is(err, ErrInvalid) {
		t.Errorf("short query: got %v, want ErrInvalid", err)
	}
}
