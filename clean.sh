#!/bin/bash

if [ $# -eq 0 ]
then
    echo "USAGE: sh clean-all.sh [FILES TO CLEAR]"
    echo "\t0: All of the below"
    echo "\t1: Target directory"
    echo "\t2: d4v2 related files"
    echo "\t3: ddnnife results"
    exit 0
fi

# target directory
if [ $1 -eq 0 ] || [ $1 -eq 1 ]
then
    echo "Removing the target directory..."
    cargo clean
fi

# d4v2 related files
if [ $1 -eq 0 ] || [ $1 -eq 2 ]
then
    echo "Removing d4v2 related files..."
    rm -rf d4v2
    rm src/parser/d4v2.bin 2> /dev/null
fi

if [ $1 -eq 0 ] || [ $1 -eq 3 ]
then
    echo "Removing ddnnife result files..."
    rm *-features.csv \
        *-queries.csv \
        *-sat.csv \
        *-atomic.csv \
        *-wise.csv \
        *-urs.csv \
        *-stream.csv \
        *-core.csv \
        *-anomalies.txt \
        *-saved.nnf \
        *-mermaid.md \
        *.config \
        tarpaulin-report.html \
        out.txt \
        out.nnf \
        out.csv 2> /dev/null
fi