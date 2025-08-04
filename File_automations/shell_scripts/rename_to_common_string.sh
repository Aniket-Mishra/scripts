#!/bin/bash

echo "Rename all files in a folder to a common string and integers"
echo "Usage: $0 <working_dir> <new_str_name>"
echo "Example: $0 ~/Downloads/images new_images_str"
echo

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/folder new_base_name"
    exit 1
fi

TARGET_DIR="$1"
BASE_NAME="$2"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

counter=1

### This one removes files - i.e. overwrites. DO NOT USE, here for doc
# for file in "$TARGET_DIR"/*; do
#     [ -f "$file" ] || continue
#     extension="${file##*.}"

#     new_name="${BASE_NAME}_${counter}.${extension}"
#     mv "$file" "$TARGET_DIR/$new_name"
#     echo "Renamed $((counter)) files"
#     ((counter++))
# done

counter=1

for file in "$TARGET_DIR"/*; do
    [ -f "$file" ] || continue

    extension="${file##*.}"

    while :; do
        new_name="${BASE_NAME}_${counter}.${extension}"
        new_path="$TARGET_DIR/$new_name"
        if [ ! -e "$new_path" ]; then
            break
        fi
        ((counter++))
    done

    mv "$file" "$new_path"
    ((counter++))
done

echo "Renamed $((counter)) files"
