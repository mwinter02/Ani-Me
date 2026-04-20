#!/usr/bin/env zsh
# setup.sh — Ani-Me project setup script
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script will:
#   1. Create a Python virtual environment (.venv)
#   2. Install all Python dependencies
#   3. Print instructions for the manual SMPL model file setup (required for ROMP)
#
# Tested on macOS with Python 3.12 (python.org installer).
# See README.md for notes on compatibility with other platforms.

set -e  # exit on error

PYTHON=python3
VENV_DIR=".venv"

# ─────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BOLD="\033[1m"
RESET="\033[0m"

info()    { echo "${GREEN}[setup]${RESET} $*"; }
warn()    { echo "${YELLOW}[setup]${RESET} $*"; }
section() { echo "\n${BOLD}$*${RESET}"; }

# ─────────────────────────────────────────────
# 1. Create virtual environment
# ─────────────────────────────────────────────
section "── Step 1: Creating virtual environment ──"
if [ -d "$VENV_DIR" ]; then
  warn ".venv already exists — skipping creation."
else
  $PYTHON -m venv "$VENV_DIR"
  info "Created .venv using $($PYTHON --version)"
fi

source "$VENV_DIR/bin/activate"
info "Activated .venv"

# Resolve the actual Python version inside the venv (e.g. "3.12")
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version in venv: $PYTHON_VERSION"


# ─────────────────────────────────────────────
# 2. Upgrade pip
# ─────────────────────────────────────────────
section "── Step 2: Upgrading pip ──"
pip install --quiet --upgrade pip
info "pip upgraded to $(pip --version | awk '{print $2}')"

# ─────────────────────────────────────────────
# 3. Install base dependencies
# ─────────────────────────────────────────────
section "── Step 3: Installing base dependencies ──"
# setuptools must be <82 — torch 2.x requires it
pip install "setuptools<82" numpy cython lapx
info "Base dependencies installed."

# ─────────────────────────────────────────────
# 4. Install PyTorch (latest, for Mac MPS support)
# ─────────────────────────────────────────────
section "── Step 4: Installing PyTorch ──"
pip install --upgrade torch torchvision
info "PyTorch installed."

# ─────────────────────────────────────────────
# 5. Install remaining requirements
# ─────────────────────────────────────────────
section "── Step 5: Installing project requirements ──"
pip install opencv-contrib-python scipy smplx
# chumpy and simple_romp both require --no-build-isolation (old build systems
# that need venv-installed pip/Cython rather than an isolated build environment)
pip install chumpy --no-build-isolation
pip install simple_romp==1.1.4 --no-build-isolation
info "All packages installed."

# ─────────────────────────────────────────────
# Patch chumpy for Python 3.11+ / NumPy 2.x compatibility
# chumpy is unmaintained and has two known breakages on modern Python:
#   1. inspect.getargspec removed in 3.11 → use getfullargspec
#   2. numpy type aliases (np.int, np.bool etc.) removed in 1.24 → remove import
# Paths are resolved from the venv's actual Python version, not hardcoded.
# ─────────────────────────────────────────────
section "── Step 5b: Patching chumpy for Python $PYTHON_VERSION / NumPy 2.x ──"
CHUMPY_CH="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/chumpy/ch.py"
CHUMPY_INIT="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/chumpy/__init__.py"

if [ -f "$CHUMPY_CH" ]; then
  sed -i '' 's/inspect\.getargspec/inspect.getfullargspec/g' "$CHUMPY_CH"
  info "Patched chumpy/ch.py (getargspec → getfullargspec)"
else
  warn "chumpy/ch.py not found at $CHUMPY_CH — skipping patch."
fi

if [ -f "$CHUMPY_INIT" ]; then
  sed -i '' \
    's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' \
    "$CHUMPY_INIT"
  info "Patched chumpy/__init__.py (removed deprecated numpy aliases)"
else
  warn "chumpy/__init__.py not found at $CHUMPY_INIT — skipping patch."
fi

# ─────────────────────────────────────────────
# Fix macOS Python SSL certificates (macOS only)
# Only applies to Python installed via python.org — Homebrew/pyenv/conda
# manage their own certs and don't need this.
# ─────────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
  section "── Step 5c: Fixing macOS Python SSL certificates ──"
  # Find the Install Certificates command for the active Python version
  CERT_CMD="/Applications/Python ${PYTHON_VERSION}/Install Certificates.command"
  if [ -f "$CERT_CMD" ]; then
    "$CERT_CMD"
    info "SSL certificates updated."
  else
    warn "Certificate installer not found at '$CERT_CMD'."
    warn "This is only needed for python.org installs — Homebrew/pyenv/conda manage certs automatically."
    warn "If you hit SSL errors later, run 'Install Certificates.command' from your Python app folder."
  fi
fi

# ─────────────────────────────────────────────
# 6. Manual setup instructions for SMPL models
# ─────────────────────────────────────────────
section "── Step 6: Manual setup required — SMPL model files ──"
cat <<'EOF'

  ROMP requires SMPL model files that cannot be distributed automatically
  due to licensing.

  1. Follow the download instructions in README.md to prepare your
     smpl_model_data/ folder.

  2. Once the folder is ready, run:

       ./prepare_smpl.sh

     If your smpl_model_data/ folder is not in the project root, pass
     the path explicitly:

       ./prepare_smpl.sh /path/to/smpl_model_data

     This script will verify all required files are present (with hints
     if anything is missing) and then convert them for ROMP.

  ──────────────────────────────────────────────────────────────────────
  To activate your environment in future sessions:
    source .venv/bin/activate

  To run the pipeline:
    python -c "from src.pipeline import run_pipeline; run_pipeline(...)"

EOF

info "Python setup complete. See above for next steps."


