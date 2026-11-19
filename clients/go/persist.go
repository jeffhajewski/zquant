package zquant

/*
#include "zquant.h"
*/
import "C"

import (
	"os"
	"runtime"
	"unsafe"
)

// The serialized form holds the configuration, any calibration and the codes, in a
// versioned, checksummed, little-endian format that is the same on every platform. A
// loaded index searches and accepts new vectors exactly as the saved one did.

// MarshalBinary serializes the index, implementing encoding.BinaryMarshaler. It runs
// alongside searches, but waits for an Add in flight, so the bytes are always a whole
// index.
func (ix *Index) MarshalBinary() ([]byte, error) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if ix.handle == nil {
		return nil, errIndexClosed
	}
	size := int(C.zq_index_serialized_size(ix.handle))
	buf := make([]byte, size)
	var written C.size_t
	// size is 0 only for an index that cannot be saved (a partitioned one), and the
	// library says why when handed no buffer.
	var ptr unsafe.Pointer
	if size > 0 {
		ptr = unsafe.Pointer(&buf[0])
	}
	st := C.zq_index_serialize(ix.handle, ptr, C.size_t(size), &written)
	runtime.KeepAlive(ix)
	if err := check(st, "serialize"); err != nil {
		return nil, err
	}
	return buf[:written], nil
}

// FromBytes loads an index serialized by MarshalBinary or Save. data is copied, so the
// caller may reuse it afterwards. Anything that is not an intact index from a
// compatible version, including truncated or corrupted bytes, gives an error matching
// ErrFormat.
func FromBytes(data []byte) (*Index, error) {
	if err := checkCPU(); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, &Error{Status: C.ZQ_ERR_FORMAT, Context: "load: no data"}
	}
	var handle *C.zq_index
	st := C.zq_index_deserialize(unsafe.Pointer(&data[0]), C.size_t(len(data)), &handle)
	if err := check(st, "load"); err != nil {
		return nil, err
	}
	return wrapIndex(handle), nil
}

// Save writes the serialized index to path, replacing any file there. A file cut short
// by a crash fails its checksum on Load rather than loading wrong.
func (ix *Index) Save(path string) error {
	data, err := ix.MarshalBinary()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// Load reads an index written by Save.
func Load(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return FromBytes(data)
}
