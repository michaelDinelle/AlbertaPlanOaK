#!/bin/zsh
set -e
cd "$(dirname "$0")"
clang -O3 -march=native -flto -ffp-contract=on -lm -DSTEPS=${STEPS:-1000000} "$@" -o oak main.c agent.c
