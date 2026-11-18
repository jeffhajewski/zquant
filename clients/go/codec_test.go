package zquant

import (
	"bytes"
	"errors"
	"math"
	"runtime"
	"sync"
	"testing"
)

// relErr is ||a-b|| / ||a|| over the whole batch.
func relErr(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		e := float64(a[i] - b[i])
		num += e * e
		den += float64(a[i]) * float64(a[i])
	}
	return math.Sqrt(num / den)
}

func TestCodecRoundTrip(t *testing.T) {
	const d, n = 128, 200
	// Not unit length, so the norms carry real information.
	x := corpus(n, d, 80, 0.3)
	for i := range x {
		x[i] *= 3
	}

	prev := math.Inf(1)
	for bits := 1; bits <= 6; bits++ {
		c, err := NewCodec(CodecConfig{Dim: d, Bits: bits, Seed: 9})
		if err != nil {
			t.Fatal(err)
		}
		if c.Dim() != d || c.CodeBytes() != d*bits/8 {
			t.Errorf("bits=%d: Dim %d, CodeBytes %d; want %d, %d", bits, c.Dim(), c.CodeBytes(), d, d*bits/8)
		}
		codes, norms, err := c.Encode(x)
		if err != nil {
			t.Fatal(err)
		}
		if len(codes) != n*c.CodeBytes() || len(norms) != n {
			t.Fatalf("bits=%d: %d code bytes and %d norms for %d vectors", bits, len(codes), len(norms), n)
		}
		for i := 0; i < n; i++ {
			var sq float64
			for _, v := range x[i*d : (i+1)*d] {
				sq += float64(v) * float64(v)
			}
			if math.Abs(float64(norms[i])-math.Sqrt(sq)) > 1e-4*math.Sqrt(sq) {
				t.Fatalf("bits=%d: norm %d is %v, want %v", bits, i, norms[i], math.Sqrt(sq))
			}
		}
		y, err := c.Decode(codes, norms)
		if err != nil {
			t.Fatal(err)
		}
		e := relErr(x, y)
		t.Logf("bits=%d: %d B/vector + norm, relative error %.4f", bits, c.CodeBytes(), e)
		if e >= prev {
			t.Errorf("bits=%d: error %.4f did not improve on %.4f", bits, e, prev)
		}
		prev = e
		c.Close()
	}
	if prev > 0.05 {
		t.Errorf("6-bit relative error %.4f is too high", prev)
	}
}

func TestCodecIsDeterministic(t *testing.T) {
	const d = 100 // not a power of two
	x := corpus(10, d, 81, 0)
	a, _ := NewCodec(CodecConfig{Dim: d, Bits: 3, Seed: 5})
	defer a.Close()
	b, _ := NewCodec(CodecConfig{Dim: d, Bits: 3, Seed: 5})
	defer b.Close()
	other, _ := NewCodec(CodecConfig{Dim: d, Bits: 3, Seed: 6})
	defer other.Close()
	ca, _, _ := a.Encode(x)
	cb, _, _ := b.Encode(x)
	co, _, _ := other.Encode(x)
	if !bytes.Equal(ca, cb) {
		t.Error("same config, different codes")
	}
	if bytes.Equal(ca, co) {
		t.Error("different seeds, same codes")
	}
}

func TestCodecErrors(t *testing.T) {
	for _, cfg := range []CodecConfig{{Dim: 0}, {Dim: MaxDim + 1}, {Dim: 8, Bits: 7}, {Dim: 8, Bits: -1}} {
		if _, err := NewCodec(cfg); !errors.Is(err, ErrInvalid) {
			t.Errorf("%+v: got %v, want ErrInvalid", cfg, err)
		}
	}
	const d = 16
	c, err := NewCodec(CodecConfig{Dim: d, Bits: 1})
	if err != nil {
		t.Fatal(err)
	}
	x := corpus(3, d, 82, 0)
	codes, norms, _ := c.Encode(x)

	if _, _, err := c.Encode(x[:d+1]); !errors.Is(err, ErrInvalid) {
		t.Errorf("ragged encode: %v", err)
	}
	nan := append([]float32(nil), x...)
	nan[3] = float32(math.NaN())
	if _, _, err := c.Encode(nan); !errors.Is(err, ErrInvalid) {
		t.Errorf("NaN encode: got %v, want ErrInvalid", err)
	}
	if _, err := c.Decode(codes[:len(codes)-1], norms); !errors.Is(err, ErrInvalid) {
		t.Errorf("short codes: %v", err)
	}
	if _, err := c.Decode(nil, nil); !errors.Is(err, ErrInvalid) {
		t.Errorf("empty decode: %v", err)
	}

	c.Close()
	c.Close()
	if _, _, err := c.Encode(x); !errors.Is(err, ErrClosed) {
		t.Errorf("encode after close: got %v, want ErrClosed", err)
	}
	if _, err := c.Decode(codes, norms); !errors.Is(err, ErrClosed) {
		t.Errorf("decode after close: got %v, want ErrClosed", err)
	}
}

// One Codec shared by several goroutines takes turns rather than corrupting its
// scratch, and Close landing among them waits for the call in flight.
func TestCodecConcurrentUseAndClose(t *testing.T) {
	const d = 64
	x := corpus(64, d, 83, 0)
	c, _ := NewCodec(CodecConfig{Dim: d, Bits: 4})
	want, wantNorms, _ := c.Encode(x)

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; ; i++ {
				codes, norms, err := c.Encode(x)
				if errors.Is(err, ErrClosed) {
					return
				}
				if err != nil {
					errs <- err
					return
				}
				if !bytes.Equal(codes, want) || norms[0] != wantNorms[0] {
					errs <- errors.New("concurrent encode produced different codes")
					return
				}
				if _, err := c.Decode(codes, norms); err != nil && !errors.Is(err, ErrClosed) {
					errs <- err
					return
				}
				// Every goroutine closes it in turn; only the first Close does anything.
				if i == 20 {
					c.Close()
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

func TestCodecFinalizer(t *testing.T) {
	const d = 32
	x := corpus(8, d, 84, 0)
	for i := 0; i < 30; i++ {
		c, err := NewCodec(CodecConfig{Dim: d})
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := c.Encode(x); err != nil {
			t.Fatal(err)
		}
		if i%5 == 0 {
			runtime.GC()
		}
	}
	runtime.GC()
}
