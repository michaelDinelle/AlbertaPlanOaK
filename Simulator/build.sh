#!/bin/zsh
set -e
cd "$(dirname "$0")"
f=(-O3 -march=native -flto -lm -DSTEPS=${STEPS:-1000000})
clang $f "$@" -o oak main.c agent.c
for t in reference separability; do clang $f -o $t $t.c; done
