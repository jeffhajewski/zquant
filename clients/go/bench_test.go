package zquant

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkSearch reports throughput through the cgo boundary on the same shape the
// native benchmarks use, so the two numbers can be compared directly and the cost of the
// binding is visible rather than assumed.
func BenchmarkSearch(b *testing.B) {
	const d, n, nq = 256, 100000, 1000
	x, q := corpus(n, d, 20, 0), corpus(nq, d, 21, 0)

	ix, err := New(Config{Dim: d, Bits: 5})
	if err != nil {
		b.Fatal(err)
	}
	defer ix.Close()

	start := time.Now()
	if err := ix.Add(x); err != nil {
		b.Fatal(err)
	}
	build := time.Since(start)
	b.Logf("%d x %d, %d B/vector -> %.0f MB against %.0f MB as float32",
		n, d, ix.BytesPerVector(), float64(n*ix.BytesPerVector())/1e6, float64(n*d*4)/1e6)
	b.Logf("build %.2fs (%.0f vectors/s)", build.Seconds(), float64(n)/build.Seconds())

	for _, threads := range []int{1, 10} {
		b.Run(fmt.Sprintf("threads=%d", threads), func(b *testing.B) {
			if _, err := ix.Search(q[:64*d], 10, threads); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := ix.Search(q, 10, threads); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(nq)*float64(b.N)/b.Elapsed().Seconds(), "QPS")
		})
	}
}
