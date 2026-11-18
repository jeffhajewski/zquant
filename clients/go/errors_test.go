package zquant

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestCPUAndBuildInfo(t *testing.T) {
	// The tests only get this far on a CPU the library runs on, so this must hold.
	if !CPUSupported() {
		t.Fatalf("CPUSupported() = false on the machine running the tests (built for %s)", BuildCPU())
	}
	if BuildCPU() == "" {
		t.Error("empty BuildCPU")
	}
	if !strings.Contains(BuildKernels(), "zquant-kernels") {
		t.Errorf("BuildKernels() = %q", BuildKernels())
	}
	t.Logf("cpu %s, %s", BuildCPU(), BuildKernels())
}

func TestSentinelsMatchStatus(t *testing.T) {
	sentinels := []*Error{ErrAlloc, ErrInvalid, ErrState, ErrUnsupported, ErrCPU, ErrFormat}
	for _, a := range sentinels {
		err := error(&Error{Status: a.Status, Context: "something"})
		for _, b := range sentinels {
			if got := errors.Is(err, b); got != (a == b) {
				t.Errorf("errors.Is(status %d, sentinel %d) = %v", a.Status, b.Status, got)
			}
		}
		if errors.Is(err, ErrClosed) {
			t.Errorf("status %d matched ErrClosed", a.Status)
		}
		if strings.Contains(a.Error(), "unknown status") {
			t.Errorf("sentinel %d has no message: %s", a.Status, a.Error())
		}
	}
	// Specific errors must not match each other through the sentinel rule.
	if errors.Is(&Error{Status: -2, Context: "a"}, &Error{Status: -2, Context: "b"}) {
		t.Error("two contextual errors matched")
	}
}

func TestValidationErrors(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"dim 0", Config{Dim: 0}},
		{"dim negative", Config{Dim: -3}},
		{"dim past MaxDim", Config{Dim: MaxDim + 1}},
		{"bits 1", Config{Dim: 8, Bits: 1}},
		{"bits 7", Config{Dim: 8, Bits: 7}},
		{"bits negative", Config{Dim: 8, Bits: -1}},
		{"metric", Config{Dim: 8, Metric: 3}},
	}
	for _, c := range cases {
		_, err := New(c.cfg)
		if !errors.Is(err, ErrInvalid) {
			t.Errorf("%s: got %v, want ErrInvalid", c.name, err)
		}
	}

	// The boundaries themselves are accepted.
	for _, cfg := range []Config{{Dim: 1, Bits: 2}, {Dim: MaxDim, Bits: 6}} {
		ix, err := New(cfg)
		if err != nil {
			t.Errorf("%+v: %v", cfg, err)
			continue
		}
		ix.Close()
	}

	ix, _ := New(Config{Dim: 8})
	defer ix.Close()
	x := corpus(10, 8, 50, 0)
	if err := ix.Add(make([]float32, 9)); !errors.Is(err, ErrInvalid) {
		t.Errorf("ragged add: %v", err)
	}
	if _, err := ix.NewSearcher(0, 1); !errors.Is(err, ErrInvalid) {
		t.Errorf("k=0: %v", err)
	}
	if _, err := ix.NewSearcher(5, MaxThreads+1); !errors.Is(err, ErrInvalid) {
		t.Errorf("too many threads: %v", err)
	}
	if err := ix.Add(x); err != nil {
		t.Fatal(err)
	}
	if err := ix.Calibrate(x); !errors.Is(err, ErrState) {
		t.Errorf("calibrate after add: got %v, want ErrState", err)
	}
}

func TestNonFiniteInputIsInvalid(t *testing.T) {
	const d = 8
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		ix, _ := New(Config{Dim: d})
		x := corpus(4, d, 51, 0)
		x[5] = bad
		if err := ix.Add(x); !errors.Is(err, ErrInvalid) {
			t.Errorf("add %v: got %v, want ErrInvalid", bad, err)
		}
		if ix.Len() != 0 {
			t.Errorf("a refused batch left %d vectors behind", ix.Len())
		}
		if err := ix.Calibrate(x); !errors.Is(err, ErrInvalid) {
			t.Errorf("calibrate %v: got %v, want ErrInvalid", bad, err)
		}
		ix.Add(corpus(4, d, 52, 0))
		if _, err := ix.Search(x[:d], 2, 1); !errors.Is(err, ErrInvalid) {
			t.Errorf("query %v: got %v, want ErrInvalid", bad, err)
		}
		ix.Close()
	}
}

func TestErrorMessagesNameTheProblem(t *testing.T) {
	_, err := New(Config{Dim: 70000})
	if err == nil || !strings.Contains(err.Error(), "70000") || !strings.Contains(err.Error(), "65536") {
		t.Errorf("dim error does not say what was wrong: %v", err)
	}
	var zerr *Error
	if !errors.As(err, &zerr) || zerr.Status != -2 {
		t.Errorf("errors.As: %v", err)
	}
}
