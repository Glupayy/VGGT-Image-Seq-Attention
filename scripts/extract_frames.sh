#!/usr/bin/env bash
# Uniformly sample image files without loading or decoding them.
set -euo pipefail

SOURCE_DIR=""
OUTPUT_DIR="./extracted_frames"
NUM_FRAMES="32"

show_usage() {
    cat <<USAGE
Usage: $0 SOURCE_DIR [options]

Sample PNG, JPG, and JPEG files recursively in filename order.

Options:
  -o, --output DIR       Output directory (default: ./extracted_frames)
  -n, --num-frames N     Positive number of frames (default: 32)
  -h, --help             Show this help
  --                     End options (for a source starting with '-')

Examples:
  $0 ./input_frames -o ./extracted_frames -n 16
  $0 "/path/to/frames with spaces" --num-frames 8
USAGE
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -o|--output)
            [ "$#" -ge 2 ] && [ -n "$2" ] && [[ "$2" != -* ]] || fail "$1 requires an output directory."
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--num-frames)
            [ "$#" -ge 2 ] && [ -n "$2" ] || fail "$1 requires a positive integer."
            NUM_FRAMES="$2"
            shift 2
            ;;
        --)
            shift
            [ "$#" -eq 1 ] && [ -z "$SOURCE_DIR" ] || fail "Specify exactly one source directory."
            SOURCE_DIR="$1"
            shift
            ;;
        -*)
            fail "Unknown option: $1. Use --help for usage."
            ;;
        *)
            [ -z "$SOURCE_DIR" ] || fail "Specify exactly one source directory."
            SOURCE_DIR="$1"
            shift
            ;;
    esac
done

[ -n "$SOURCE_DIR" ] || fail "A source directory is required. Use --help for usage."
[[ "$NUM_FRAMES" =~ ^[0-9]+$ ]] || fail "Frame count must be a positive integer."
# Strip leading zeros before shell arithmetic (which otherwise treats them as octal).
NUM_FRAMES=$(printf '%s' "$NUM_FRAMES" | sed 's/^0*//')
[ -n "$NUM_FRAMES" ] || fail "Frame count must be a positive integer."
# Keep arithmetic portable and reject integer overflow before evaluating it.
if [ "${#NUM_FRAMES}" -gt 10 ] || { [ "${#NUM_FRAMES}" -eq 10 ] && [[ "$NUM_FRAMES" > 2147483647 ]]; }; then
    fail "Frame count must be at most 2147483647."
fi
[ -d "$SOURCE_DIR" ] || fail "Source directory does not exist: $SOURCE_DIR"
SOURCE_DIR=$(cd -- "$SOURCE_DIR" && pwd -P)

temp_file=$(mktemp)
trap 'rm -f "$temp_file"' EXIT
# Null delimiters preserve spaces and newlines in image filenames.
find "$SOURCE_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | LC_ALL=C sort -z > "$temp_file"
files=()
while IFS= read -r -d '' image; do
    files+=("$image")
done < "$temp_file"
total_files=${#files[@]}
[ "$total_files" -gt 0 ] || fail "No PNG, JPG, or JPEG files found in: $SOURCE_DIR"

if [ "$total_files" -lt "$NUM_FRAMES" ]; then
    printf 'Found %s images; sampling all available frames.\n' "$total_files"
    NUM_FRAMES=$total_files
fi
mkdir -p -- "$OUTPUT_DIR"

for ((i = 0; i < NUM_FRAMES; i++)); do
    if [ "$NUM_FRAMES" -eq 1 ]; then
        index=$((total_files / 2))
    else
        index=$((i * (total_files - 1) / (NUM_FRAMES - 1)))
    fi
    source_file=${files[$index]}
    filename=${source_file##*/}
    printf -v prefix 'frame_%03d_' "$i"
    cp -- "$source_file" "$OUTPUT_DIR/$prefix$filename"
    printf 'Copied: %s\n' "$prefix$filename"
done

printf 'Saved %s frames to: %s\n' "$NUM_FRAMES" "$OUTPUT_DIR"
