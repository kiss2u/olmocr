#!/bin/bash

datasets=(
    "s2orcforolmo-unspecified-oa"
)

for dataset in "${datasets[@]}"; do
    python -m olmocr.pipeline "s3://ai2-oe-data/jakep/dolma4pdfs_workspaces/${dataset}" \
        --pdfs "s3://ai2-oe-data/jakep/dolma4pdfs_frontier/${dataset}/*.gz" \
        --workers 2 --beaker --beaker_gpus 64 --beaker_priority high
done
