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

// CodecConfig describes a Codec. Dim is required.
type CodecConfig struct {
	Dim int
	// Bits is the codebook width per coordinate, 1..6. Unlike the index there is no
	// residual sketch reserving a bit, so Bits here buys more accuracy than the same
	// number in Config. Zero means 5.
	Bits int
	// Seed picks the random rotation. Zero means 0x5EED, as for Config. Encoder and
	// decoder must use the same seed.
	Seed uint64
}

// Codec compresses vectors and gives them back, without an index.
//
// The index answers "which stored vectors best match this query". A KV cache asks
// something else: store these vectors compactly and return them. Attention needs every
// score, accurate in absolute terms, and the index's estimator is fitted to preserve
// ranking instead; reconstruction through a Codec measured considerably better there.
//
// Each vector becomes CodeBytes bytes of bit-packed code plus one float32 norm, and
// decoding needs both. A Codec owns scratch space, so calls on one Codec take turns;
// for parallel encoding give each goroutine its own. Close waits for a call in flight,
// is idempotent, and any call after it returns an error wrapping ErrClosed.
type Codec struct {
	// mu guards handle and is held across every native call that uses it.
	mu        sync.Mutex
	handle    *C.zq_codec
	dim       int
	codeBytes int
}

// errCodecClosed is what every call on a closed codec returns.
const errCodecClosed = closedError("codec")

// NewCodec creates a codec.
func NewCodec(cfg CodecConfig) (*Codec, error) {
	if cfg.Dim < 1 || cfg.Dim > MaxDim {
		return nil, invalid("Dim %d is outside 1..%d", cfg.Dim, MaxDim)
	}
	bits := cfg.Bits
	if bits == 0 {
		bits = 5
	}
	if bits < 1 || bits > 6 {
		return nil, invalid("Bits %d is outside 1..6", cfg.Bits)
	}
	if err := checkCPU(); err != nil {
		return nil, err
	}
	seed := cfg.Seed
	if seed == 0 {
		seed = 0x5EED
	}
	c := C.zq_codec_config{dim: C.uint32_t(cfg.Dim), bits: C.uint8_t(bits), seed: C.uint64_t(seed)}
	var handle *C.zq_codec
	if err := check(C.zq_codec_create(&c, &handle), "codec"); err != nil {
		return nil, err
	}
	codec := &Codec{handle: handle, dim: cfg.Dim, codeBytes: int(C.zq_codec_code_bytes(handle))}
	runtime.SetFinalizer(codec, (*Codec).Close)
	return codec, nil
}

// Close releases the codec. It is idempotent.
func (c *Codec) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle != nil {
		C.zq_codec_free(c.handle)
		c.handle = nil
		runtime.SetFinalizer(c, nil)
	}
	return nil
}

// Dim reports the vector dimension.
func (c *Codec) Dim() int { return c.dim }

// CodeBytes reports the packed code size of one vector, not counting its norm.
func (c *Codec) CodeBytes() int { return c.codeBytes }

// Encode compresses row-major vectors, len(vectors) being a multiple of Dim. Vector i's
// code is codes[i*CodeBytes : (i+1)*CodeBytes] and its norm is norms[i].
func (c *Codec) Encode(vectors []float32) (codes []byte, norms []float32, err error) {
	if len(vectors) == 0 || len(vectors)%c.dim != 0 {
		return nil, nil, invalid("vectors length %d is not a positive multiple of dim %d", len(vectors), c.dim)
	}
	n := len(vectors) / c.dim
	codes = make([]byte, n*c.codeBytes)
	norms = make([]float32, n)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle == nil {
		return nil, nil, errCodecClosed
	}
	st := C.zq_codec_encode(c.handle, (*C.float)(&vectors[0]), C.size_t(n),
		(*C.uint8_t)(unsafe.Pointer(&codes[0])), (*C.float)(&norms[0]))
	runtime.KeepAlive(c)
	if err := check(st, "encode"); err != nil {
		return nil, nil, err
	}
	return codes, norms, nil
}

// Decode reconstructs vectors from codes and norms as Encode returned them, one norm
// per vector. It returns len(norms)*Dim floats, row-major.
func (c *Codec) Decode(codes []byte, norms []float32) ([]float32, error) {
	n := len(norms)
	if n == 0 {
		return nil, invalid("no vectors to decode")
	}
	if len(codes) != n*c.codeBytes {
		return nil, invalid("codes length %d is not %d norms times %d code bytes", len(codes), n, c.codeBytes)
	}
	out := make([]float32, n*c.dim)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle == nil {
		return nil, errCodecClosed
	}
	st := C.zq_codec_decode(c.handle, (*C.uint8_t)(unsafe.Pointer(&codes[0])),
		(*C.float)(&norms[0]), C.size_t(n), (*C.float)(&out[0]))
	runtime.KeepAlive(c)
	if err := check(st, "decode"); err != nil {
		return nil, err
	}
	return out, nil
}
