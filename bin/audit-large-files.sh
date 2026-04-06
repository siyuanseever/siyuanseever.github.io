#!/usr/bin/env bash

set -euo pipefail

# Inspect current tracked files and historical blobs that are larger than the
# provided thresholds. This helps separate "current site payload" from "Git
# history baggage".

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

current_threshold_mb="${1:-1}"
history_threshold_mb="${2:-5}"

current_threshold_bytes=$((current_threshold_mb * 1024 * 1024))
history_threshold_bytes=$((history_threshold_mb * 1024 * 1024))

echo "Repository: $repo_root"
echo "Git dir size:"
du -sh .git
echo

echo "Git object summary:"
git count-objects -vH
echo

echo "Current tracked files >= ${current_threshold_mb} MiB:"
git ls-files | while IFS= read -r file; do
  size=$(wc -c < "$file" 2>/dev/null || true)
  if [[ -n "${size}" && "${size}" -ge "${current_threshold_bytes}" ]]; then
    printf '%12s %s\n' "$size" "$file"
  fi
done | sort -n
echo

echo "Historical blobs >= ${history_threshold_mb} MiB:"
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk -v limit="$history_threshold_bytes" '$1 == "blob" && $3 >= limit { print }' \
  | sort -k3 -n
