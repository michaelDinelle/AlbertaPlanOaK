#!/bin/zsh
set -e
cd "$(dirname "$0")"
AGENT=${AGENT:-oak}
SOURCES=(main.c jbw.c "agent_$AGENT.c")
FLAGS=()
if [[ "$AGENT" == "scent" ]]; then
    FLAGS+=(-DWITH_SCENT)
fi
clang -O3 -mcpu=apple-m4 -flto -ffp-contract=on -lm -DSTEPS=${STEPS:-1000000} \
    $FLAGS "$@" -o "$AGENT" $SOURCES
