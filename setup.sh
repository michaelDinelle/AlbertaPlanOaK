#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

echo "OaK 2026 Setup"

VENV_DIR="${OAK_VENV_DIR:-$ROOT_DIR/.venv}"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
PYTHON_IMPORT_CHECK='import gymnasium, minigrid, PyFlyt, dm_control, mujoco, envpool, pybullet'

# 1. Locate Python 3.11 (Homebrew: brew install python@3.11)
PY_BOOTSTRAP=""
if command -v python3.11 >/dev/null 2>&1; then
    PY_BOOTSTRAP="python3.11"
elif command -v python3 >/dev/null 2>&1; then
    if python3 -c 'import sys; exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then
        PY_BOOTSTRAP="python3"
    fi
fi
[ -n "$PY_BOOTSTRAP" ] || fail "Python 3.11 not found on PATH. Install e.g.: brew install python@3.11"

DIRECT_PYTHON="$VENV_DIR/bin/python"
[ -f "$REQUIREMENTS_FILE" ] || fail "Missing requirements file at $REQUIREMENTS_FILE"

# 2. Create a clean virtual environment
echo "Creating Python virtual environment at $VENV_DIR..."
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi
"$PY_BOOTSTRAP" -m venv "$VENV_DIR"
[ -x "$DIRECT_PYTHON" ] || fail "Python executable not found at $DIRECT_PYTHON"

# 3. Install dependencies (pybullet builds from sdist on macOS — requires Xcode Command Line Tools)
echo "Installing Python packages..."
"$DIRECT_PYTHON" -m pip install --upgrade pip setuptools wheel

# PyBullet: no Apple wheels on PyPI; sdist vendors zlib whose fdopen macro breaks the SDK when
# TARGET_OS_MAC is set. Patch that block, then install from source before the rest of requirements.
PB_VER="3.2.7"
OAK_PB_CACHE="${OAK_PB_CACHE:-$ROOT_DIR/.oak-setup-cache}"
echo "Preparing pybullet ${PB_VER} (zlib fdopen patch)..."
mkdir -p "$OAK_PB_CACHE"
PB_SRC="$OAK_PB_CACHE/pybullet-$PB_VER"
if [ ! -f "$PB_SRC/setup.py" ]; then
    "$DIRECT_PYTHON" -m pip download --no-binary=:all: "pybullet==${PB_VER}" -d "$OAK_PB_CACHE"
    tar -xzf "$OAK_PB_CACHE/pybullet-${PB_VER}.tar.gz" -C "$OAK_PB_CACHE"
fi
ZUTIL="$PB_SRC/examples/ThirdPartyLibs/zlib/zutil.h"
[ -f "$ZUTIL" ] || fail "pybullet extract missing $ZUTIL"
"$DIRECT_PYTHON" - "$ZUTIL" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "/* pybullet-oak: macOS fdopen vendored-zlib patch */"
if marker in text:
    sys.exit(0)
old = """#else
#ifndef fdopen
#define fdopen(fd, mode) NULL /* No fdopen() */
#endif
#endif
#endif
#endif

#ifdef TOPS20"""
new = """#else
#ifndef fdopen
#if !defined(__APPLE__)
#define fdopen(fd, mode) NULL /* No fdopen() */
#endif
#endif
#endif
#endif
#endif

#ifdef TOPS20"""
if old not in text:
    sys.exit("pybullet zutil.h: expected MACOS/fdopen block not found (pybullet version mismatch?)")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
"$DIRECT_PYTHON" -m pip install "$PB_SRC"

"$DIRECT_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

echo "Validating Python environment..."
"$DIRECT_PYTHON" -c "$PYTHON_IMPORT_CHECK" >/dev/null || {
    echo "EnvPool often needs Homebrew Qt 5: brew install qt@5" >&2
    fail "Python import validation failed"
}
echo "✓ Python environment validated"

# 4. Build OaK
echo ""
echo "Building OaK..."
swift build -c release

PYTHON_LIB_HINT="$("$DIRECT_PYTHON" -c 'import os, sys, sysconfig
ldl = sysconfig.get_config_var("LDLIBRARY") or ""
libdir = sysconfig.get_config_var("LIBDIR") or ""
if not ldl:
    raise SystemExit("Could not resolve LDLIBRARY for PYTHON_LIBRARY hint")
candidates = [
    os.path.normpath(os.path.join(libdir, ldl)),
    os.path.normpath(os.path.join(sys.base_prefix, ldl)),
    os.path.normpath(os.path.join(sys.base_prefix, "Python")),
]
for path in candidates:
    if path and os.path.isfile(path):
        print(path)
        break
else:
    raise SystemExit("Could not locate libpython/Python for PYTHON_LIBRARY hint")
')"

PYTHONHOME_HINT="$("$DIRECT_PYTHON" -c 'import sys; print(sys.base_prefix)')"
PYTHONPATH_HINT="$("$DIRECT_PYTHON" -c 'import site; p = site.getsitepackages(); print(p[0] if p else "")')"

echo ""
echo "✓ Setup complete"
echo ""
echo "To run (from repo root):"
echo "  export OAK_ROOT_DIR=\"\$(pwd)\""
echo "  export PYTHONHOME=\"$PYTHONHOME_HINT\""
echo "  export PYTHONPATH=\"$PYTHONPATH_HINT\""
echo "  export PYTHON_LIBRARY=\"$PYTHON_LIB_HINT\""
echo "  swift run -c release -- minigrid"
