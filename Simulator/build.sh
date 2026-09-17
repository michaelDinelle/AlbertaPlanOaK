#!/bin/zsh
set -e
cd "$(dirname "$0")"
flags=(-O3 -march=native -lm)
clang $flags -DSTEPS=${STEPS:-1000000} "$@" -o oak main.c agent.c
clang $flags -DSTEPS=${STEPS:-1000000} -o reference reference.c
clang $flags -o separability separability.c
