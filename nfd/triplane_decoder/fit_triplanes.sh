#!/bin/bash
OFFDIR=$1
PTSDIR=$2
TRIDIR=$3

echo "OFFDIR: $OFFDIR"
echo "PTSDIR: $PTSDIR"
echo "TRIDIR: $TRIDIR"

# conda activate 3d_mlp

for FILE in $OFFDIR/*
do
    BASENAME=${FILE##*/}
    BASENAME=${BASENAME%%.*}

    if test -f "$TRIDIR/$BASENAME.npy"; then
        echo "Skipping $TRIDIR/$BASENAME.npy"
    else
        echo Preprocessing $FILE...
        python generate_3d_dataset.py --input $FILE --output $PTSDIR/$BASENAME.npy --count 10000000
        python fit_single.py --input $PTSDIR/$BASENAME.npy --output $TRIDIR/$BASENAME.npy
        rm -rf $PTSDIR/$BASENAME.npy
    fi
done
