#!/bin/bash

# Take in a path to a s3://ai2-oe-data/jakep/dolma4pdfs_workspaces/s2orcforolmo-CCBY_workspace/work_index_list.csv.zstd
# Download and extract that
# Replace /lustre/orion/csc652/scratch/jakep/dolma4pdfs/s2orcforolmo-CCBY/s2orcforolmo-CCBY_worker787_batch000066.tar.gz
# with s3://ai2-oe-data/jakep/dolma4pdfs_frontier/s2orcforolmo-CCBY/s2orcforolmo-CCBY_worker787_batch000066.tar.gz
# Move s3://ai2-oe-data/jakep/dolma4pdfs_workspaces/s2orcforolmo-CCBY_workspace/work_index_list.csv.zstd to s3://ai2-oe-data/jakep/dolma4pdfs_workspaces/s2orcforolmo-CCBY_workspace/work_index_list.csv.zstd.bak
# Write the replaced file in the original location

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <s3-path-to-work_index_list.csv.zstd>"
    echo "Example: $0 s3://ai2-oe-data/jakep/dolma4pdfs_workspaces/s2orcforolmo-CCBY_workspace/work_index_list.csv.zstd"
    exit 1
fi

S3_PATH="$1"

# Create a temporary directory for working files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

ZSTD_FILE="$TMPDIR/work_index_list.csv.zstd"
CSV_FILE="$TMPDIR/work_index_list.csv"
MODIFIED_CSV="$TMPDIR/work_index_list_modified.csv"
MODIFIED_ZSTD="$TMPDIR/work_index_list_modified.csv.zstd"

echo "Downloading $S3_PATH..."
aws s3 cp "$S3_PATH" "$ZSTD_FILE"

echo "Decompressing..."
zstd -d "$ZSTD_FILE" -o "$CSV_FILE"

echo "Converting Lustre paths to S3 paths..."
# Replace /lustre/orion/csc652/scratch/jakep/dolma4pdfs/ with s3://ai2-oe-data/jakep/dolma4pdfs_frontier/
sed 's|/lustre/orion/csc652/scratch/jakep/dolma4pdfs/|s3://ai2-oe-data/jakep/dolma4pdfs_frontier/|g' "$CSV_FILE" > "$MODIFIED_CSV"

echo "Compressing modified file..."
zstd "$MODIFIED_CSV" -o "$MODIFIED_ZSTD"

echo "Backing up original file to ${S3_PATH}.bak..."
aws s3 mv "$S3_PATH" "${S3_PATH}.bak"

echo "Uploading modified file..."
aws s3 cp "$MODIFIED_ZSTD" "$S3_PATH"

echo "Done! Original backed up to ${S3_PATH}.bak"