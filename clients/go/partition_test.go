package zquant

import (
	"errors"
	"testing"
)

func TestPartition(t *testing.T) {
	const d, n, k = 48, 4000, 10
	x, q := corpus(n, d, 70, 0.3), corpus(50, d, 71, 0.3)
	exact := exactTopK(x, q, d, k)

	ix, _ := New(Config{Dim: d, Bits: 4})
	defer ix.Close()
	if ix.Lists() != 0 {
		t.Fatal("Lists before Partition")
	}
	// A cached searcher from before partitioning must not be reused after it.
	if err := ix.Add(x[:d]); err != nil {
		t.Fatal(err)
	}
	if err := ix.Partition(x, 40, 2); !errors.Is(err, ErrState) {
		t.Fatalf("partition after add: got %v, want ErrState", err)
	}

	ix2, _ := New(Config{Dim: d, Bits: 4})
	defer ix2.Close()
	early, err := ix2.NewSearcher(k, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer early.Close()
	if err := early.SetProbes(3); !errors.Is(err, ErrState) {
		t.Fatalf("SetProbes on a flat index: got %v, want ErrState", err)
	}
	if err := ix2.Partition(x, 0, 1); !errors.Is(err, ErrInvalid) {
		t.Fatalf("0 lists: got %v, want ErrInvalid", err)
	}
	if err := ix2.Partition(x, 40, 2); err != nil {
		t.Fatal(err)
	}
	if ix2.Lists() != 40 {
		t.Fatalf("Lists %d, want 40", ix2.Lists())
	}
	if err := ix2.Calibrate(x); !errors.Is(err, ErrState) {
		t.Fatalf("calibrate after partition: got %v, want ErrState", err)
	}
	if err := ix2.AddParallel(x, 3); err != nil {
		t.Fatal(err)
	}
	if _, err := ix2.SearchWith(early, q); !errors.Is(err, ErrState) {
		t.Fatalf("searcher from before partitioning: got %v, want ErrState", err)
	}

	s, err := ix2.NewSearcher(k, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	recallAt := func(probes int) float64 {
		if err := s.SetProbes(probes); err != nil {
			t.Fatal(err)
		}
		r, err := ix2.SearchWith(s, q)
		if err != nil {
			t.Fatal(err)
		}
		hits := 0
		for i := range exact {
			ids, _ := r.At(i)
			for _, want := range exact[i] {
				for _, got := range ids {
					if int(got) == want {
						hits++
					}
				}
			}
		}
		return float64(hits) / float64(len(exact)*k)
	}
	one, all := recallAt(1), recallAt(40)
	if !(one < all) || all < 0.8 {
		t.Fatalf("recall with 1 probe %.3f, with every list %.3f", one, all)
	}
	// Search uses the default, an eighth of the lists.
	def, err := ix2.Search(q, k, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetProbes(5); err != nil {
		t.Fatal(err)
	}
	five, _ := ix2.SearchWith(s, q)
	for i := range def.IDs {
		if def.IDs[i] != five.IDs[i] {
			t.Fatal("Search does not probe an eighth of the lists")
		}
	}

	// ScoreIDs reports what a search did, for ids in any list.
	ids, scores := five.At(0)
	got, err := ix2.ScoreIDs(s, q[:d], ids)
	if err != nil {
		t.Fatal(err)
	}
	for j := range ids {
		if got[j] != scores[j] {
			t.Fatalf("id %d: ScoreIDs %v, search %v", ids[j], got[j], scores[j])
		}
	}
	if _, err := ix2.ScoreIDs(early, q[:d], ids[:1]); !errors.Is(err, ErrState) {
		t.Fatalf("ScoreIDs with a searcher from before partitioning: got %v, want ErrState", err)
	}

	// A saved partitioned index loads partitioned, and searches as the original does.
	data, err := ix2.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	back, err := FromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	defer back.Close()
	if back.Lists() != 40 || back.Len() != ix2.Len() || back.BytesPerVector() != ix2.BytesPerVector() {
		t.Fatalf("loaded: %d lists, %d vectors, %d bytes each", back.Lists(), back.Len(), back.BytesPerVector())
	}
	bs, err := back.NewSearcher(k, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer bs.Close()
	if err := bs.SetProbes(5); err != nil {
		t.Fatal(err)
	}
	loaded, err := back.SearchWith(bs, q)
	if err != nil {
		t.Fatal(err)
	}
	for i := range five.IDs {
		if five.IDs[i] != loaded.IDs[i] || five.Scores[i] != loaded.Scores[i] {
			t.Fatal("the loaded index searches differently")
		}
	}
	// A searcher may outlive its index; setting probes on it is still safe.
	ix2.Close()
	if err := s.SetProbes(2); err != nil {
		t.Fatalf("SetProbes after the index closed: %v", err)
	}
}
