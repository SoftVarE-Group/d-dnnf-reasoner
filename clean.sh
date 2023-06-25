#!/bin/bash

if [ $# -eq 0 ]
then
    echo "USAGE: sh clean-all.sh [FILES TO CLEAR]"
    echo -e "\t0: All of the below"
    echo -e "\t1: Target directory"
    echo -e "\t2: d4v2 related files"
    echo -e "\t3: ddnnife results"
    echo -e "\t4: ddnnife image"
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
    rm src/bin/d4v2.bin 2> /dev/null
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

if [ $1 -eq 0 ] || [ $1 -eq 4 ]
then
    echo "Removing ddnnife image..."
    docker image rm ddnnife &> /dev/null
fi