package zquant

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io/fs"
	"path/filepath"
	"testing"
)

var _ encoding.BinaryMarshaler = (*Index)(nil)

func TestConfigReadsBackFromALoadedIndex(t *testing.T) {
	const d = 40
	x := corpus(600, d, 73, 0.5)
	for _, cfg := range []Config{
		{Dim: d, Bits: 3, Metric: InnerProduct, Seed: 0x5EED},
		{Dim: d, Bits: 3, Metric: Cosine, Seed: 7, Expanded: true},
		{Dim: d, Bits: 3, Metric: L2, Seed: ^uint64(0)},
	} {
		ix, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if got, err := ix.Config(); err != nil || got != cfg {
			t.Fatalf("new index: Config() = %+v, %v; want %+v", got, err, cfg)
		}
		if ix.Calibrated() {
			t.Fatal("fresh index reports calibration")
		}
		if err := ix.Calibrate(x); err != nil {
			t.Fatal(err)
		}
		ix.Add(x)
		data, _ := ix.MarshalBinary()
		ix.Close()

		back, err := FromBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		got, err := back.Config()
		if err != nil || got != cfg || !back.Calibrated() {
			t.Fatalf("loaded: Config() = %+v, %v, calibrated %v; want %+v", got, err, back.Calibrated(), cfg)
		}
		// The configuration is enough to build an index that encodes identically.
		twin, err := New(got)
		if err != nil {
			t.Fatal(err)
		}
		twin.Calibrate(x)
		twin.Add(x)
		if twinData, _ := twin.MarshalBinary(); !bytes.Equal(twinData, data) {
			t.Fatalf("%+v: an index built from Config() encodes differently", cfg)
		}
		twin.Close()
		back.Close()
		if back.Calibrated() {
			t.Fatal("closed index reports calibration")
		}
	}

	// Defaults are reported as the values in effect.
	ix, _ := New(Config{Dim: 8})
	defer ix.Close()
	if got, _ := ix.Config(); got.Bits != 5 || got.Seed != 0x5EED {
		t.Fatalf("defaults read back as Bits %d Seed %#x", got.Bits, got.Seed)
	}
}

func TestConfigRefusesWhatItCannotDescribe(t *testing.T) {
	ix, _ := New(Config{Dim: 16})
	ix.Add(corpus(20, 16, 74, 0))
	data, _ := ix.MarshalBinary()
	ix.Close()
	data[22] = 1 // the exact scan, which only the Zig API can set
	crc := crc32.ChecksumIEEE(data[:len(data)-4])
	binary.LittleEndian.PutUint32(data[len(data)-4:], crc)
	back, err := FromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	defer back.Close()
	if back.Len() != 20 {
		t.Fatalf("Len %d", back.Len())
	}
	if _, err := back.Config(); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("Config() error %v, want ErrUnsupported", err)
	}
}

func TestRoundTrip(t *testing.T) {
	const d = 96
	x, more, q := corpus(800, d, 70, 0.5), corpus(100, d, 71, 0.5), corpus(20, d, 72, 0.5)
	configs := []struct {
		cfg       Config
		calibrate bool
	}{
		{Config{Dim: d, Bits: 4, Metric: InnerProduct, Seed: 1}, false},
		{Config{Dim: d, Bits: 5, Metric: Cosine, Seed: 2}, false},
		{Config{Dim: d, Bits: 3, Metric: L2, Seed: 3, Expanded: true}, false},
		{Config{Dim: d, Bits: 4, Metric: InnerProduct, Seed: 4}, true},
	}
	for _, c := range configs {
		orig, err := New(c.cfg)
		if err != nil {
			t.Fatal(err)
		}
		if c.calibrate {
			if err := orig.Calibrate(x); err != nil {
				t.Fatalf("%+v: %v", c.cfg, err)
			}
		}
		orig.Add(x)
		data, err := orig.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		loaded, err := FromBytes(data)
		if err != nil {
			t.Fatalf("%+v: %v", c.cfg, err)
		}
		if loaded.Len() != orig.Len() || loaded.Dim() != d || loaded.BytesPerVector() != orig.BytesPerVector() {
			t.Fatalf("%+v: loaded Len %d Dim %d BPV %d", c.cfg, loaded.Len(), loaded.Dim(), loaded.BytesPerVector())
		}
		again, _ := loaded.MarshalBinary()
		if !bytes.Equal(data, again) {
			t.Fatalf("%+v: re-serializing a loaded index changed its bytes", c.cfg)
		}

		// It searches as the original did, and keeps accepting vectors the same way.
		a, _ := orig.Search(q, 10, 1)
		b, _ := loaded.Search(q, 10, 1)
		for i := range a.IDs {
			if a.IDs[i] != b.IDs[i] || a.Scores[i] != b.Scores[i] {
				t.Fatalf("%+v: loaded index searches differently at %d", c.cfg, i)
			}
		}
		orig.Add(more)
		loaded.Add(more)
		x1, _ := orig.MarshalBinary()
		x2, _ := loaded.MarshalBinary()
		if !bytes.Equal(x1, x2) {
			t.Fatalf("%+v: adding to the loaded index diverged from the original", c.cfg)
		}
		orig.Close()
		loaded.Close()
	}
}

func TestAddParallelIsByteIdentical(t *testing.T) {
	const d = 64
	x := corpus(4000, d, 73, 0)
	a, _ := New(Config{Dim: d})
	defer a.Close()
	b, _ := New(Config{Dim: d})
	defer b.Close()
	a.Add(x)
	b.AddParallel(x, 8)
	ab, _ := a.MarshalBinary()
	bb, _ := b.MarshalBinary()
	if !bytes.Equal(ab, bb) {
		t.Fatal("AddParallel built a different index from Add")
	}
}

func TestSaveLoad(t *testing.T) {
	const d = 32
	ix, _ := New(Config{Dim: d})
	defer ix.Close()
	ix.Add(corpus(200, d, 74, 0))
	path := filepath.Join(t.TempDir(), "index.zq")
	if err := ix.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	want, _ := ix.MarshalBinary()
	got, _ := loaded.MarshalBinary()
	if !bytes.Equal(want, got) {
		t.Fatal("Load gave a different index from the one saved")
	}

	if _, err := Load(filepath.Join(t.TempDir(), "missing")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("missing file: %v", err)
	}
	ix2, _ := New(Config{Dim: d})
	ix2.Close()
	if err := ix2.Save(path); !errors.Is(err, ErrClosed) {
		t.Errorf("save after close: got %v, want ErrClosed", err)
	}
}

func TestDamagedBytesAreFormatErrors(t *testing.T) {
	const d = 32
	ix, _ := New(Config{Dim: d})
	defer ix.Close()
	ix.Add(corpus(50, d, 75, 0))
	data, _ := ix.MarshalBinary()

	bad := map[string][]byte{
		"empty":     {},
		"garbage":   []byte("this is not a zquant index, not even close"),
		"truncated": data[:len(data)-1],
		"header":    data[:8],
	}
	for _, pos := range []int{0, len(data) / 2, len(data) - 1} {
		flipped := bytes.Clone(data)
		flipped[pos] ^= 0x40
		bad[fmt.Sprintf("byte %d flipped", pos)] = flipped
	}
	for name, b := range bad {
		loaded, err := FromBytes(b)
		if !errors.Is(err, ErrFormat) {
			t.Errorf("%s: got %v, want ErrFormat", name, err)
		}
		if loaded != nil {
			loaded.Close()
		}
	}
}
