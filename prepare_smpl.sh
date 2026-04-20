#!/usr/bin/env zsh
# prepare_smpl.sh — Convert SMPL model files for use with ROMP
#
# Usage:
#   chmod +x prepare_smpl.sh
#   ./prepare_smpl.sh [/path/to/smpl_model_data]
#
# If no path is given, defaults to ./smpl_model_data (the folder name
# produced when unzipping smpl_model_data.zip).
#
# See README.md for download links and instructions on how to prepare
# the smpl_model_data folder before running this script.

set -e

# ─────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BOLD="\033[1m"
RESET="\033[0m"

info()  { echo "${GREEN}[prepare_smpl]${RESET} $*"; }
warn()  { echo "${YELLOW}[prepare_smpl]${RESET} $*"; }
error() { echo "${RED}[prepare_smpl] ERROR:${RESET} $*"; }

# ─────────────────────────────────────────────
# Resolve source directory — default to ./smpl_model_data
# ─────────────────────────────────────────────
SOURCE_DIR="${1:-./smpl_model_data}"

# ─────────────────────────────────────────────
# Check directory exists
# ─────────────────────────────────────────────
if [ ! -d "$SOURCE_DIR" ]; then
  error "Directory not found: $SOURCE_DIR"
  echo ""
  if [ -z "$1" ]; then
    echo "  No path was provided, so the default './smpl_model_data' was used."
    echo "  Make sure you have unzipped smpl_model_data.zip in this directory."
    echo "  The zip should unpack directly to a folder named 'smpl_model_data/'."
  else
    echo "  Check that the path is correct and the folder is named 'smpl_model_data'."
  fi
  echo "  See README.md for download instructions."
  exit 1
fi

info "Found directory: $SOURCE_DIR"

# ─────────────────────────────────────────────
# Check required files
# ─────────────────────────────────────────────
REQUIRED_FILES=(
  "SMPL_NEUTRAL.pkl"
  "J_regressor_extra.npy"
  "J_regressor_h36m.npy"
  "smpl_kid_template.npy"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$SOURCE_DIR/$f" ]; then
    warn "Missing: $SOURCE_DIR/$f"
    MISSING=$((MISSING + 1))
  else
    info "Found: $f"
  fi
done

# Provide specific guidance for the most common missing file
if [ ! -f "$SOURCE_DIR/SMPL_NEUTRAL.pkl" ]; then
  echo ""
  echo "  ${YELLOW}Hint:${RESET} SMPL_NEUTRAL.pkl is missing."
  echo "  Download the SMPL model files from the link in README.md."
  echo "  After unzipping, locate:"
  echo "    smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
  echo "  Copy it into '$SOURCE_DIR/' and rename it to SMPL_NEUTRAL.pkl"
  echo ""
fi

if [ "$MISSING" -gt 0 ]; then
  error "$MISSING required file(s) missing from $SOURCE_DIR — see hints above."
  echo "  Refer to README.md for full download instructions."
  exit 1
fi

# ─────────────────────────────────────────────
# Activate venv if present
# ─────────────────────────────────────────────
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
  info "Activated .venv"
else
  warn "No .venv found — make sure simple_romp is installed and run setup.sh first."
fi

# ─────────────────────────────────────────────
# Pre-download ROMP.pkl via curl (avoids Python SSL issues on macOS)
# ─────────────────────────────────────────────
ROMP_DIR="$HOME/.romp"
ROMP_PKL="$ROMP_DIR/ROMP.pkl"
ROMP_PKL_URL="https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl"

mkdir -p "$ROMP_DIR"

if [ -f "$ROMP_PKL" ]; then
  info "ROMP.pkl already exists at $ROMP_PKL — skipping download."
else
  info "Downloading ROMP.pkl via curl..."
  curl -L --progress-bar "$ROMP_PKL_URL" -o "$ROMP_PKL"
  info "Downloaded ROMP.pkl to $ROMP_PKL"
fi

# ─────────────────────────────────────────────
# Run ROMP model conversion
# ─────────────────────────────────────────────
echo ""
info "Converting SMPL model files for ROMP..."
romp.prepare_smpl -source_dir="$SOURCE_DIR"

# ─────────────────────────────────────────────
# Verify output
# ─────────────────────────────────────────────
ROMP_DIR="$HOME/.romp"
EXPECTED_OUTPUT=(
  "SMPL_NEUTRAL.pth"
  "SMPLA_NEUTRAL.pth"
)

echo ""
info "Checking output in $ROMP_DIR ..."
OUTPUT_MISSING=0
for f in "${EXPECTED_OUTPUT[@]}"; do
  if [ -f "$ROMP_DIR/$f" ]; then
    info "✓ $ROMP_DIR/$f"
  else
    warn "✗ $ROMP_DIR/$f not found — conversion may have partially failed."
    OUTPUT_MISSING=$((OUTPUT_MISSING + 1))
  fi
done

echo ""
if [ "$OUTPUT_MISSING" -eq 0 ]; then
  info "SMPL model files prepared successfully. You are ready to run the pipeline."
else
  warn "Some output files are missing. Check the output above for errors."
  warn "This may still be okay — ROMP sometimes reports errors but succeeds regardless."
fi

