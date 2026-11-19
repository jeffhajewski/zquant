// Package zquant provides a flat quantized vector index backed by TurboQuant.
//
//	index, err := zquant.New(zquant.Config{Dim: 256, Bits: 5})
//	defer index.Close()
//	index.Add(vectors)                        // len(vectors) == n*Dim
//	res, err := index.Search(queries, 10, 8)  // k=10, 8 threads
//
// Vectors are float32, row-major, one per row.
//
// Beyond that: SearchRerank and ScoreIDs for exact reranking and scoring chosen ids,
// AddParallel for multi-threaded encoding, Save, Load and MarshalBinary for
// persistence, and Codec for encoding and decoding vectors without an index. Every
// error is a *Error that errors.Is matches against one of the Err sentinels.
package zquant

/*
#include <stdlib.h>
#include "zquant.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

// Metric selects how a stored vector is scored against a query.
type Metric int

const (
	InnerProduct Metric = C.ZQ_METRIC_INNER_PRODUCT
	Cosine       Metric = C.ZQ_METRIC_COSINE
	L2           Metric = C.ZQ_METRIC_L2
)

// Error is a failure, carrying the native status code it corresponds to. Arguments the
// binding rejects before calling the library carry the status the library would have
// returned, so one errors.Is check covers both.
type Error struct {
	Status  int
	Context string
}

func (e *Error) Error() string {
	msg := C.GoString(C.zq_status_string(C.int(e.Status)))
	if e.Context == "" {
		return fmt.Sprintf("zquant: %s (%d)", msg, e.Status)
	}
	return fmt.Sprintf("zquant: %s: %s (%d)", e.Context, msg, e.Status)
}

// Is reports whether target is the sentinel for e's status, so that
// errors.Is(err, zquant.ErrInvalid) matches every invalid-argument error whatever its
// context.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Context == "" && t.Status == e.Status
}

// Sentinels for each native status, for use with errors.Is. Every *Error returned by
// this package matches exactly one of them.
var (
	ErrAlloc       = &Error{Status: C.ZQ_ERR_ALLOC}       // out of memory
	ErrInvalid     = &Error{Status: C.ZQ_ERR_INVALID}     // bad argument, including NaN or infinite input
	ErrState       = &Error{Status: C.ZQ_ERR_STATE}       // not valid now, e.g. Calibrate after Add
	ErrUnsupported = &Error{Status: C.ZQ_ERR_UNSUPPORTED} // a configuration the library cannot build
	ErrCPU         = &Error{Status: C.ZQ_ERR_CPU}         // this CPU cannot run this build; see CPUSupported
	ErrFormat      = &Error{Status: C.ZQ_ERR_FORMAT}      // not an intact index from a compatible version
)

func check(status C.int, context string) error {
	if status == C.ZQ_OK {
		return nil
	}
	return &Error{Status: int(status), Context: context}
}

// invalid is an argument error caught before it reaches the library.
func invalid(format string, args ...any) error {
	return &Error{Status: C.ZQ_ERR_INVALID, Context: fmt.Sprintf(format, args...)}
}

// ErrClosed is returned, wrapped, by any call on an Index, Searcher or Codec after its
// Close.
var ErrClosed = errors.New("zquant: use after Close")

// closedError names what was closed while still matching ErrClosed under errors.Is.
type closedError string

func (e closedError) Error() string        { return "zquant: " + string(e) + " is closed" }
func (e closedError) Is(target error) bool { return target == ErrClosed }

// Limits the library enforces. The binding checks them first, so the error can say
// which argument is wrong.
const (
	MaxDim     = int(C.ZQ_MAX_DIM)     // largest vector dimension
	MaxThreads = int(C.ZQ_MAX_THREADS) // most threads one call may use
)

var cpu struct {
	once sync.Once
	ok   bool
}

// CPUSupported reports whether this CPU can run the linked build of libzquant.
//
// Release builds use instructions beyond the architecture baseline (SSSE3 on x86_64,
// the dot-product extension on arm64), because search is 6-10x slower without them. On
// a CPU that lacks them every constructor in this package returns an error matching
// ErrCPU rather than letting the library die of an illegal instruction; the fix is to
// link a portable build. The check itself is safe on any CPU, and runs once.
func CPUSupported() bool {
	cpu.once.Do(func() { cpu.ok = C.zq_cpu_supported() != 0 })
	return cpu.ok
}

// BuildCPU reports the CPU model the linked library was compiled for, such as
// "x86_64_v2", for diagnostics.
func BuildCPU() string { return C.GoString(C.zq_build_cpu()) }

// BuildKernels reports which scan kernels the linked library uses, such as
// "zquant-kernels shuffle=native dot=native". "scalar" means the portable fallback,
// which is correct but several times slower.
func BuildKernels() string { return C.GoString(C.zq_build_kernels()) }

// checkCPU is the guard every constructor runs before anything else native.
func checkCPU() error {
	if CPUSupported() {
		return nil
	}
	return &Error{
		Status:  C.ZQ_ERR_CPU,
		Context: fmt.Sprintf("libzquant was built for %s (%s) and this CPU lacks instructions it needs; link a portable build", BuildCPU(), BuildKernels()),
	}
}

// Version reports the native library version.
func Version() string { return C.GoString(C.zq_version()) }

// Config describes an index. The zero value is not usable; Dim is required.
type Config struct {
	Dim int
	// Bits is the total budget per coordinate, 2..6. The scalar codebook uses Bits-1.
	// Zero means 5, which is a good default: 132 B/vector at R@10 0.916 on 256-d
	// embeddings.
	Bits   int
	Metric Metric
	Seed   uint64
	// Expanded stores dequantized int8 rather than packed codes: roughly twice the
	// memory for some throughput.
	Expanded bool
}

// indexIDs hands each Index a unique identity, so a Searcher can tell whether it is
// being used with the index it was made for without holding a pointer back to it.
var indexIDs atomic.Uint64

// Index is a flat (exhaustive) quantized index. Every vector is scanned for every
// query, so recall is bounded only by quantization and cost grows linearly with the
// corpus.
//
// Every method is safe to call from any goroutine. Searches run in parallel with each
// other; Add and Calibrate wait for searches in flight and hold new ones off until they
// finish, because the native library forbids a mutation overlapping any other call on
// the same index. Close waits for every call in flight, and any call after it returns
// an error wrapping ErrClosed.
type Index struct {
	// mu guards handle. Calls that only read the index (search, count, serialize,
	// creating a searcher) hold it shared; calls that modify it, and Close, hold it
	// exclusively. Holding it across each native call is what stops Close freeing the
	// handle under a search.
	mu     sync.RWMutex
	handle *C.zq_index
	dim    int
	id     uint64

	// cfg is read back from the library once, so a loaded index reports what it was
	// saved with. cfgErr is set instead for an index saved through the Zig API with
	// options Config has no field for.
	cfg    Config
	cfgErr error

	// cacheMu guards cache. It is only taken with mu already held, shared or
	// exclusive, so the lock order is always mu then cacheMu.
	cacheMu sync.Mutex
	cache   map[[2]int]*Searcher
}

// New creates an empty index.
func New(cfg Config) (*Index, error) {
	if cfg.Dim < 1 || cfg.Dim > MaxDim {
		return nil, invalid("Dim %d is outside 1..%d", cfg.Dim, MaxDim)
	}
	bits := cfg.Bits
	if bits == 0 {
		bits = 5
	}
	if bits < 2 || bits > 6 {
		return nil, invalid("Bits %d is outside 2..6", cfg.Bits)
	}
	switch cfg.Metric {
	case InnerProduct, Cosine, L2:
	default:
		return nil, invalid("unknown Metric %d", cfg.Metric)
	}
	if err := checkCPU(); err != nil {
		return nil, err
	}
	seed := cfg.Seed
	if seed == 0 {
		seed = 0x5EED
	}
	compact := C.int(1)
	if cfg.Expanded {
		compact = 0
	}
	c := C.zq_config{
		dim:     C.uint32_t(cfg.Dim),
		bits:    C.uint8_t(bits),
		metric:  C.int(cfg.Metric),
		seed:    C.uint64_t(seed),
		compact: compact,
	}
	var handle *C.zq_index
	if err := check(C.zq_index_create(&c, &handle), "create"); err != nil {
		return nil, err
	}
	return wrapIndex(handle), nil
}

// wrapIndex takes ownership of a native index handle.
func wrapIndex(handle *C.zq_index) *Index {
	ix := &Index{
		handle: handle,
		dim:    int(C.zq_index_dim(handle)),
		id:     indexIDs.Add(1),
		cache:  map[[2]int]*Searcher{},
	}
	var c C.zq_config
	if ix.cfgErr = check(C.zq_index_config(handle, &c), "Config"); ix.cfgErr == nil {
		ix.cfg = Config{
			Dim:      int(c.dim),
			Bits:     int(c.bits),
			Metric:   Metric(c.metric),
			Seed:     uint64(c.seed),
			Expanded: c.compact == 0,
		}
	}
	// A finalizer is a backstop, not the contract: callers should Close. Without it a
	// dropped Index leaks native memory the garbage collector cannot see. Nothing the
	// Index reaches points back at it, so it never sits in a cycle, which would stop
	// the finalizer from running at all.
	runtime.SetFinalizer(ix, (*Index).Close)
	return ix
}

// Close releases the index and the searchers Search cached for it. It waits for calls
// in flight to finish, and is idempotent.
//
// Searchers made with NewSearcher are not closed here; close them yourself. Once the
// index is closed they can no longer search, but releasing them is still safe.
func (ix *Index) Close() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ix.handle == nil {
		return nil
	}
	// No other call can hold mu, so none is using a cached searcher.
	ix.cacheMu.Lock()
	for _, s := range ix.cache {
		s.Close()
	}
	ix.cache = nil
	ix.cacheMu.Unlock()

	C.zq_index_free(ix.handle)
	ix.handle = nil
	runtime.SetFinalizer(ix, nil)
	return nil
}

// errIndexClosed is what every call on a closed index returns.
const errIndexClosed = closedError("index")

// Dim reports the vector dimension.
func (ix *Index) Dim() int { return ix.dim }

// Config reports the configuration the index was created with, or saved with if it was
// loaded, for when all you have is a file. Passing it to New gives an index that
// encodes identically, apart from calibration (see Calibrated). An index saved through
// the Zig API with options Config cannot describe gives an error matching
// ErrUnsupported.
//
// Bits and Seed are the values in effect, never the zero that New reads as "the
// default". One consequence: another binding can create an index whose seed really is
// 0, and since New turns a zero Seed into the default, New cannot recreate that one.
func (ix *Index) Config() (Config, error) { return ix.cfg, ix.cfgErr }

// Calibrated reports whether Calibrate was applied, here or before the index was
// saved. It reports false once the index is closed.
func (ix *Index) Calibrated() bool {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if ix.handle == nil {
		return false
	}
	calibrated := C.zq_index_calibrated(ix.handle) != 0
	runtime.KeepAlive(ix)
	return calibrated
}

// Len reports how many vectors the index holds, or 0 once it is closed.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.lenLocked()
}

func (ix *Index) lenLocked() int {
	if ix.handle == nil {
		return 0
	}
	n := int(C.zq_index_count(ix.handle))
	runtime.KeepAlive(ix)
	return n
}

// BytesPerVector reports storage per vector, codes and scalars together, or 0 once the
// index is closed.
func (ix *Index) BytesPerVector() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if ix.handle == nil {
		return 0
	}
	n := int(C.zq_index_bytes_per_vector(ix.handle))
	runtime.KeepAlive(ix)
	return n
}

// rows checks that v holds a whole, non-zero number of vectors and returns how many.
func (ix *Index) rows(v []float32, name string) (int, error) {
	if len(v) == 0 || len(v)%ix.dim != 0 {
		return 0, invalid("%s length %d is not a positive multiple of dim %d", name, len(v), ix.dim)
	}
	return len(v) / ix.dim, nil
}

// Calibrate fits the corpus centre from a sample. It must precede Add.
//
// Each vector is then stored relative to the centre, and the centre's share of every
// score is computed exactly, for 2 more bytes per vector. Worth doing when the corpus
// sits away from the origin — the norm of the mean of the unit-normalized vectors. At
// 0.65 and above it gained 9 to 22 points of recall@10 at bits 2 and 3; at 0.12, about
// a point. A few hundred rows are enough.
func (ix *Index) Calibrate(sample []float32) error {
	n, err := ix.rows(sample, "sample")
	if err != nil {
		return err
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ix.handle == nil {
		return errIndexClosed
	}
	st := C.zq_index_calibrate(ix.handle, (*C.float)(&sample[0]), C.size_t(n))
	runtime.KeepAlive(ix)
	return check(st, "calibrate")
}

// Add appends vectors to the index. Ids are assigned in insertion order, from 0.
func (ix *Index) Add(vectors []float32) error {
	n, err := ix.rows(vectors, "vectors")
	if err != nil {
		return err
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ix.handle == nil {
		return errIndexClosed
	}
	st := C.zq_index_add(ix.handle, (*C.float)(&vectors[0]), C.size_t(n))
	runtime.KeepAlive(ix)
	return check(st, "add")
}

// AddParallel is Add with encoding spread over up to threads threads; 0 or 1 encodes
// on the calling goroutine. The index ends up byte-for-byte the same as after Add.
// Below a few hundred vectors per thread the library encodes on one thread anyway,
// since starting a thread costs more than that much work.
func (ix *Index) AddParallel(vectors []float32, threads int) error {
	n, err := ix.rows(vectors, "vectors")
	if err != nil {
		return err
	}
	if threads < 0 || threads > MaxThreads {
		return invalid("threads %d is outside 0..%d", threads, MaxThreads)
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ix.handle == nil {
		return errIndexClosed
	}
	st := C.zq_index_add_parallel(ix.handle, (*C.float)(&vectors[0]), C.size_t(n), C.size_t(threads))
	runtime.KeepAlive(ix)
	return check(st, "add")
}

// CalibrateAuto measures the sample's centroid norm and calibrates only if it is at
// least 0.3; below that, calibration gained about a point, not worth its 2 bytes per
// vector. It returns the norm and whether calibration was applied, and fails exactly as
// Calibrate would, whatever the measurement. Like Calibrate it must precede Add.
//
// This is the sensible default when you do not know your data.
func (ix *Index) CalibrateAuto(sample []float32) (centroidNorm float32, applied bool, err error) {
	n, err := ix.rows(sample, "sample")
	if err != nil {
		return 0, false, err
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ix.handle == nil {
		return 0, false, errIndexClosed
	}
	var norm C.float
	var did C.int
	st := C.zq_index_calibrate_auto(ix.handle, (*C.float)(&sample[0]), C.size_t(n), &norm, &did)
	runtime.KeepAlive(ix)
	if err := check(st, "calibrate"); err != nil {
		return 0, false, err
	}
	return float32(norm), did != 0, nil
}
