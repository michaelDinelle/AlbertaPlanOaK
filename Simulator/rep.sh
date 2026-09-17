#!/bin/zsh
awk '/^  mean reward/{r=$3} /^  eaten /{g=$2;b=$5;c=$7} /^  %forward/{f=$2} END{printf "%s  eaten %4d/%-4d %-8s fwd %s\n",r,g,b,c,f}' $1
