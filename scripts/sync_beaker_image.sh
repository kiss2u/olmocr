#!/bin/bash

set -e

VERSION=$(python -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "Syncing olmOCR version $VERSION with model included"

# Pull the image with model included
docker pull alleninstituteforai/olmocr:v$VERSION-with-model

# Create beaker image with the model included version
beaker image create --workspace ai2/oe-data-pdf --name olmocr-inference-$VERSION alleninstituteforai/olmocr:v$VERSION-with-model

echo "Successfully synced olmocr:v$VERSION-with-model to beaker as olmocr-inference-$VERSION"
