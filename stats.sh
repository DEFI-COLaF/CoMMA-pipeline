#!/bin/bash

echo "Generating file extension statistics (ignoring .manifest.json)..."

find . -type d | while read -r dir; do
    [ "$dir" = "." ] && continue

    # Include hidden files
    shopt -s nullglob dotglob
    files=("$dir"/*)
    shopt -u nullglob dotglob

    echo "📁 $dir — file counts by extension (excluding .manifest.json):"
    declare -A ext_count
    any_files=false

    for f in "${files[@]}"; do
        [ -f "$f" ] || continue
        filename=$(basename "$f")

        # Skip .manifest.json
        [[ "$filename" == ".manifest.json" ]] && continue

        ext="${filename##*.}"
        [[ "$filename" == "$ext" ]] && ext="[no extension]"
        ((ext_count["$ext"]++))
        any_files=true
    done

    if [ "$any_files" = true ]; then
        for ext in "${!ext_count[@]}"; do
            echo "    .$ext: ${ext_count[$ext]}"
        done
    else
        echo "    (no files excluding .manifest.json)"
    fi

    unset ext_count
    echo
done

