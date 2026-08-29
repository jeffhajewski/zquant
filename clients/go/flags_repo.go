//go:build repolocal

// Repository-local: link directly against the build tree, for working on zquant itself.
//
//	zig build lib
//	go test -tags repolocal ./...
//
// The rpath is what lets the resulting test binary find the shared library at run time
// without LD_LIBRARY_PATH or DYLD_LIBRARY_PATH being set by hand.
package zquant

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo darwin LDFLAGS: -L${SRCDIR}/../../zig-out/lib -lzquant -Wl,-rpath,${SRCDIR}/../../zig-out/lib
#cgo linux LDFLAGS: -L${SRCDIR}/../../zig-out/lib -lzquant -Wl,-rpath,${SRCDIR}/../../zig-out/lib
#cgo windows LDFLAGS: -L${SRCDIR}/../../zig-out/lib -lzquant
*/
import "C"
