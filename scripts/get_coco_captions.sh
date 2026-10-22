#!/usr/bin/env bash
# Download the MS-COCO 2017 caption annotations into data/coco/annotations/.
# The S3 URL is used because images.cocodataset.org fails the TLS hostname check.
set -euo pipefail

dest="${1:-data/coco}"
url="https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"

mkdir -p "$dest"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL -o "$tmp/annotations.zip" "$url"
unzip -o -q "$tmp/annotations.zip" 'annotations/captions_*' -d "$dest"
ls -l "$dest/annotations"
