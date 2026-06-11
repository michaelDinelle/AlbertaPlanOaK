#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p .build
xcrun -sdk macosx metal -fno-fast-math -c Sources/OakKernel.metal -o .build/OakKernel.air
xcrun -sdk macosx metallib .build/OakKernel.air -o .build/OakKernel.metallib
swift build -c release

export OAK_BENCH_ENV="${OAK_BENCH_ENV:-HalfCheetah-v4}"
export OAK_BENCH_STEPS="${OAK_REBUILD_STEPS:-10000000}"
export OAK_BENCH_SEED_COUNT="${OAK_BENCH_SEED_COUNT:-128}"
export OAK_BENCH_POLICIES="${OAK_BENCH_POLICIES:-oak,zero,random}"
export OAK_BENCH_HEARTBEAT_STEPS="${OAK_BENCH_HEARTBEAT_STEPS:-200000}"
export OAK_METRICS_STORAGE="${OAK_METRICS_STORAGE:-$ROOT/.oak-benchmarks/halfcheetah-rebuild-10m}"

exec ./.build/release/OaK benchmark
