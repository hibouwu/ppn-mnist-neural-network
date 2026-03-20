#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
download_root="${project_root}/tiny-imagenet"
archive_name="tiny-imagenet-200.zip"
archive_path="${download_root}/${archive_name}"
extract_dir="${download_root}/tiny-imagenet-200"
dataset_url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"

mkdir -p "${download_root}"
cd "${download_root}"

download_file() {
    local url="$1"
    local output="$2"

    if command -v wget >/dev/null 2>&1; then
        wget -O "${output}" "${url}"
        return
    fi

    if command -v curl >/dev/null 2>&1; then
        curl -L "${url}" -o "${output}"
        return
    fi

    echo "Error: neither wget nor curl is installed." >&2
    exit 1
}

if [ -d "${extract_dir}" ] && [ -f "${extract_dir}/wnids.txt" ]; then
    echo "Tiny-ImageNet already extracted at:"
    echo "  ${extract_dir}"
    exit 0
fi

if [ ! -f "${archive_path}" ]; then
    echo "Downloading Tiny-ImageNet..."
    download_file "${dataset_url}" "${archive_path}"
else
    echo "Using existing archive:"
    echo "  ${archive_path}"
fi

echo "Extracting archive..."
unzip -q -o "${archive_path}"

if [ ! -f "${extract_dir}/wnids.txt" ]; then
    echo "Error: extracted dataset is incomplete:"
    echo "  ${extract_dir}"
    exit 1
fi

echo "Tiny-ImageNet is ready:"
echo "  ${extract_dir}"
echo
echo "Example:"
echo "  build/ppn_train --dataset tiny-imagenet --data_dir ${extract_dir} --model cnn --epochs 1 --batch_size 64"
