package zquant

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// The cross-language conformance fixture. Every binding replays it through its public
// API and must reproduce each hash exactly; the Zig suite checks the same hashes
// against the library directly, so a mismatch here is a bug in this binding.
const fixturePath = "../../tests/golden/conformance.json"

type conformanceFixture struct {
	About           string `json:"about"`
	Input           string `json:"input"`
	IndexHash       string `json:"index_hash"`
	CodecHash       string `json:"codec_hash"`
	EncodingVersion int    `json:"encoding_version"`
	Index           []struct {
		Dim       int    `json:"dim"`
		Bits      int    `json:"bits"`
		Metric    string `json:"metric"`
		Seed      string `json:"seed"`
		Compact   bool   `json:"compact"`
		Calibrate bool   `json:"calibrate"`
		Lists     int    `json:"lists"`
		Rows      int    `json:"rows"`
		Salt      int    `json:"salt"`
		SHA256    string `json:"sha256"`
	} `json:"index"`
	Codec []struct {
		Dim    int    `json:"dim"`
		Bits   int    `json:"bits"`
		Seed   string `json:"seed"`
		Rows   int    `json:"rows"`
		Salt   int    `json:"salt"`
		SHA256 string `json:"sha256"`
	} `json:"codec"`
}

func loadFixture(t *testing.T) conformanceFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.FromSlash(fixturePath))
	if err != nil {
		t.Fatalf("reading the conformance fixture: %v", err)
	}
	var f conformanceFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("parsing the conformance fixture: %v", err)
	}
	if len(f.Index) == 0 || len(f.Codec) == 0 {
		t.Fatal("the conformance fixture has no cases")
	}
	t.Logf("about: %s", f.About)
	t.Logf("input: %s", f.Input)
	return f
}

// fixtureInput builds the fixture's input:
// x[r][j] = ((r*7919 + j*104729 + salt) mod 2001 - 1000) / 1024, as float32.
// Integer arithmetic, then one exact division by a power of two, so every language
// produces bit-identical floats.
func fixtureInput(rows, dim, salt int) []float32 {
	out := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		for j := 0; j < dim; j++ {
			k := (int64(r)*7919 + int64(j)*104729 + int64(salt)) % 2001
			out[r*dim+j] = float32(k-1000) / 1024
		}
	}
	return out
}

// parseSeed reads a hex seed. Some exceed 2^53, so they must not pass through float64.
func parseSeed(t *testing.T, s string) uint64 {
	t.Helper()
	v, err := strconv.ParseUint(s, 0, 64)
	if err != nil {
		t.Fatalf("seed %q: %v", s, err)
	}
	return v
}

func TestConformanceIndex(t *testing.T) {
	f := loadFixture(t)
	t.Logf("index_hash: %s", f.IndexHash)
	metrics := map[string]Metric{"inner_product": InnerProduct, "cosine": Cosine, "l2": L2}
	for i, c := range f.Index {
		name := fmt.Sprintf("%d/dim=%d,bits=%d,%s", i, c.Dim, c.Bits, c.Metric)
		t.Run(name, func(t *testing.T) {
			metric, ok := metrics[c.Metric]
			if !ok {
				t.Fatalf("unknown metric %q", c.Metric)
			}
			seed := parseSeed(t, c.Seed)
			if seed == 0 {
				// Config reads a zero seed as "use the default"; the fixture would need
				// a way to say 0 that this API does not have.
				t.Fatal("the fixture uses seed 0, which Config cannot express")
			}
			ix, err := New(Config{Dim: c.Dim, Bits: c.Bits, Metric: metric, Seed: seed, Expanded: !c.Compact})
			if err != nil {
				t.Fatal(err)
			}
			defer ix.Close()
			x := fixtureInput(c.Rows, c.Dim, c.Salt)
			if c.Calibrate {
				if err := ix.Calibrate(x); err != nil {
					t.Fatal(err)
				}
			}
			if c.Lists > 0 {
				if err := ix.Partition(x, c.Lists, 1); err != nil {
					t.Fatal(err)
				}
			}
			if err := ix.Add(x); err != nil {
				t.Fatal(err)
			}
			data, err := ix.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(data)
			if got := hex.EncodeToString(sum[:]); got != c.SHA256 {
				t.Fatalf("sha256 %s, want %s (%d bytes)", got, c.SHA256, len(data))
			}

			// What was saved must load back to the same bytes.
			loaded, err := FromBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			defer loaded.Close()
			again, err := loaded.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			if sum2 := sha256.Sum256(again); sum2 != sum {
				t.Fatal("the loaded index re-serializes to different bytes")
			}
		})
	}
}

func TestConformanceCodec(t *testing.T) {
	f := loadFixture(t)
	t.Logf("codec_hash: %s", f.CodecHash)
	for i, c := range f.Codec {
		name := fmt.Sprintf("%d/dim=%d,bits=%d", i, c.Dim, c.Bits)
		t.Run(name, func(t *testing.T) {
			seed := parseSeed(t, c.Seed)
			if seed == 0 {
				t.Fatal("the fixture uses seed 0, which CodecConfig cannot express")
			}
			codec, err := NewCodec(CodecConfig{Dim: c.Dim, Bits: c.Bits, Seed: seed})
			if err != nil {
				t.Fatal(err)
			}
			defer codec.Close()
			codes, norms, err := codec.Encode(fixtureInput(c.Rows, c.Dim, c.Salt))
			if err != nil {
				t.Fatal(err)
			}
			h := sha256.New()
			h.Write(codes)
			buf := make([]byte, 4)
			for _, v := range norms {
				binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
				h.Write(buf)
			}
			if got := hex.EncodeToString(h.Sum(nil)); got != c.SHA256 {
				t.Fatalf("sha256 %s, want %s", got, c.SHA256)
			}
		})
	}
}
