# Work carried on `local` that is not in recursive-oss/rec

`local` = upstream release tag `v0.1.0` (`e53663b`) + the `topic/*` branches below, merged with
`--no-ff`. `main` mirrors `recursive-oss/rec` `main` (fast-forward only; it is 5 commits ahead of
v0.1.0: Anthropic `/v1/messages`, request-seeded tool-call IDs, smoke-model RAM, README; none of
them is on `local`). Each topic branch is one change on top of `v0.1.0`.

| Topic branch | What | Upstream | Kind |
|---|---|---|---|
| `topic/host-mlx-lm-0.32` | `rec_mlx` runs on mlx-lm 0.32.x instead of the pinned 0.31.3: stop sequences + text state machine, `_serve_single(request, stream)`, worker failure reported as before; `pyproject` pin `mlx-lm>=0.32.2,<0.33`; tests | not submitted (eejd/py-rec#1). Upstream pins 0.31.3 on purpose | behaviour-preserving adaptation to a newer mlx-lm API |
| `topic/go-gopath-godebug` | `//go:debug default=go1.25` in `cli/main.go`: a build without `go.mod` (GOPATH mode) gets Go 1.20 GODEBUG defaults, under which rec's control server (`GET /status` patterns) never matches | not submitted (eejd/macports-ports-local#1473) | packaging-only |
| `topic/updater-build-switch` | `-X main.updateDisabled=<message>`: `rec update` prints the message instead of running a downloaded installer, `rec doctor` skips the release lookup; empty = upstream behaviour | not submitted (eejd/macports-ports-local#1473) | packaging-only |
| `topic/native-build-mlx-prefix` | `engine/native/build.py` reads the MLX headers/libmlx from `MLX_PREFIX` and the compiler from `CXX` | not submitted (eejd/macports-ports-local#1473) | packaging-only |

Not changed (reported only): `rec serve` answers HTTP 400 to `stream_options` other than exactly
`{"include_usage": true}` and drops `ignore_eos` / `min_tokens` (eejd/py-rec#2).

Tags: `v0.1.0-local01` (superseded: tests only, see below) and `v0.1.0-local02` (current;
local01 + stricter tests of `topic/host-mlx-lm-0.32`, no change to non-test code).

`topic/host-mlx-lm-0.32` does **not** merge onto fork `main` today (CONFLICT in
`engine/src/rec_mlx/scheduler.py`: upstream's "Derive tool call IDs from the request seed" commit
changes the adjacent `GenerationContext(...)` line). The next re-cut onto a release that contains
those commits needs that one manual resolution (keep `seeded_tool_parser(sampling, ids)` and the
new context fields); the three packaging topics merge onto `main` cleanly.
