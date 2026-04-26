# OaK

Big-World, Alberta-Plan agent on Apple Silicon.

## Setup

```bash
./setup.sh
```

## Run

```bash
swift run -c release -- run <env-id>
./benchmark.sh
```

For live benchmark diagnostics, lower the heartbeat stride and watch the JSON
stream:

```bash
OAK_BENCH_HEARTBEAT_STEPS=10000 ./benchmark.sh
python3 tools/bench_watch.py --policy oak
```

For paired candidate screens, run only the diagnostic seeds first and compare
against the current keeper:

```bash
OAK_BENCH_SEEDS=0,1,3,8 OAK_BENCH_STEPS=50000 OAK_BENCH_POLICIES=oak ./benchmark.sh
python3 tools/bench_compare.py .oak-benchmarks/grounded-deadrefresh-50k-20260426 <candidate-root>
```
