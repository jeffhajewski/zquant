//go:build !repolocal

// Default: resolve the library through pkg-config, so an installed zquant can be
// consumed without downstream builds hardcoding paths.
//
//	zig build install --prefix /usr/local
//	PKG_CONFIG_PATH=/usr/local/lib/pkgconfig go test ./...
//
// Building against a repository checkout instead is `-tags repolocal`.
package zquant

/*
#cgo pkg-config: zquant
*/
import "C"
