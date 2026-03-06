#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Download weekly tracking data from a Google Drive folder into datasets/Weeks-data.

Usage:
  bash scripts/download_data.sh "<GOOGLE_DRIVE_FOLDER_URL>"

Or set:
  export BSA_WEEKS_GDRIVE_URL="<GOOGLE_DRIVE_FOLDER_URL>"
  bash scripts/download_data.sh

Notes:
  - Requires gdown: pip install gdown
  - The script expects files named week1.csv ... weekN.csv (or zipped archives containing them)
EOF
}

if ! command -v gdown >/dev/null 2>&1; then
  echo "Missing dependency: gdown"
  echo "Install with: pip install gdown"
  exit 1
fi

URL="${1:-${BSA_WEEKS_GDRIVE_URL:-}}"
if [[ -z "${URL}" ]]; then
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/datasets/Weeks-data"
TMP_DIR="${REPO_ROOT}/datasets/.tmp_weeks_download"

mkdir -p "${TARGET_DIR}"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

echo "Downloading from Google Drive folder..."
gdown --folder --fuzzy "${URL}" -O "${TMP_DIR}"

echo "Expanding zip files (if any)..."
while IFS= read -r -d '' zip_file; do
  unzip -o -q "${zip_file}" -d "${TMP_DIR}"
done < <(find "${TMP_DIR}" -type f -iname '*.zip' -print0)

mapfile -d '' WEEK_FILES < <(find "${TMP_DIR}" -type f -iname 'week*.csv' -print0)
if [[ "${#WEEK_FILES[@]}" -eq 0 ]]; then
  echo "No week*.csv files found in downloaded folder."
  echo "Downloaded content is in: ${TMP_DIR}"
  exit 1
fi

COPIED=0
for week_file in "${WEEK_FILES[@]}"; do
  base_name="$(basename "${week_file}")"
  lower_name="$(echo "${base_name}" | tr '[:upper:]' '[:lower:]')"
  cp -f "${week_file}" "${TARGET_DIR}/${lower_name}"
  COPIED=$((COPIED + 1))
done

rm -rf "${TMP_DIR}"

echo "Copied ${COPIED} week files to: ${TARGET_DIR}"
echo "You can now run:"
echo "python3 \"Data Cleaning + Engineering/scripts/build_gravity_dataset.py\""
