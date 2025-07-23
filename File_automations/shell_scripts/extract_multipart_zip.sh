#!/bin/bash

echo "Extract Multipart ZIPs Script"
echo "Usage: $0 <working_dir> <zip_filename_substring>"
echo "Example: $0 ~/Downloads/my_zips 2_Study"
echo

WORKDIR="$1"
SUBSTRING="$2"

if [ -z "$WORKDIR" ] || [ -z "$SUBSTRING" ]; then
    echo "Missing arguments."
    echo "Please provide:"
    echo "    1. Path to folder containing zip chunks"
    echo "    2. Substring to match zip files"
    exit 1
fi

cd "$WORKDIR" || { echo "Directory not found: $WORKDIR"; exit 1; }

for f in *"$SUBSTRING"*.zip; do
    if [ -f "$f" ]; then
        echo "Extracting $f..."
        7z x "$f"
    fi
done

echo
echo "Done. Files extracted in: $WORKDIR"

