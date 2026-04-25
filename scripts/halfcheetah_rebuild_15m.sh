#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p .build
xcrun -sdk macosx metal -fno-fast-math -c Sources/OakKernel.metal -o .build/OakKernel.air
xcrun -sdk macosx metallib .build/OakKernel.air -o .build/OakKernel.metallib
swift build -c release

export OAK_BENCH_ENV="${OAK_BENCH_ENV:-HalfCheetah-v4}"
export OAK_BENCH_STEPS="${OAK_REBUILD_STEPS:-1500000}"
export OAK_BENCH_SEED_COUNT="${OAK_BENCH_SEED_COUNT:-10}"
export OAK_BENCH_POLICIES="${OAK_BENCH_POLICIES:-oak,random}"
export OAK_BENCH_CONCURRENCY="${OAK_BENCH_CONCURRENCY:-8}"
export OAK_HEARTBEAT_STEPS="${OAK_HEARTBEAT_STEPS:-100000}"
export OAK_METRICS_STORAGE="${OAK_METRICS_STORAGE:-$ROOT/.oak-benchmarks/halfcheetah-rebuild-15m}"

exec ./.build/release/OaK benchmark
