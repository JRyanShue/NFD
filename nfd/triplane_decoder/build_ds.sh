#!/bin/bash
INDIR=$1
OUTDIR=$2

echo "INDIR: $INDIR"
echo "OUTDIR: $OUTDIR"

# conda activate 3d_mlp

for FILE in $INDIR/*
do
    BASENAME=${FILE##*/}
    echo "$BASENAME"
    BASENAME=${BASENAME%%.*}


    if test -f "$OUTDIR/$BASENAME.npy"; then
        echo "Skipping $OUTDIR/$BASENAME.npy"
    else
        echo Preprocessing $FILE...
        python generate_3d_dataset.py --input $FILE --output $OUTDIR/$BASENAME.npy --count 250000
    fi
done
