// Upstream builds in module mode, where go.mod's `go 1.25.0` selects the GODEBUG
// defaults of Go 1.25. A build without go.mod (GOPATH mode, as a distribution that
// builds offline from vendored sources uses) would default to Go 1.20's, and with them
// httpmuxgo121=1: the control server's method patterns ("GET /status") then never match,
// so `rec status`, `rec stop` and the running-instance check of `rec serve` fail with
// "control request failed". Keep this equal to the `go` line of go.mod.
//go:debug default=go1.25

package main

import (
	"os"

	"charm.land/lipgloss/v2"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		_, _ = lipgloss.Fprintln(os.Stderr, failureStyle.Bold(true).Render("rec:"), err)
		os.Exit(1)
	}
}
