#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "OaK 2026 Setup - 7 Environments"

# 1. Install Miniforge if missing
CONDA_DIR="$HOME/miniforge3"
CONDA_BIN="$CONDA_DIR/bin/conda"

if [ ! -f "$CONDA_BIN" ]; then
    echo "Installing Miniforge3..."
    rm -rf "$CONDA_DIR"
    curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
    rm /tmp/miniforge.sh
fi

# 2. Setup conda for this script
export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 3. Create clean environment
echo "Creating oak environment..."
conda env remove -n oak -y 2>/dev/null || true
conda create -n oak python=3.11 -y
conda activate oak

# 4. Install pybullet from conda-forge (pre-built ARM64)
echo "Installing pybullet..."
conda install -c conda-forge pybullet -y

# 5. Install all Python packages in one go
echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel

pip install \
    "numpy==1.26.4" \
    "scipy" \
    "matplotlib" \
    "typing-extensions>=4.3.0" \
    "tqdm" \
    "requests" \
    "protobuf>=3.19.4" \
    "lxml" \
    "pyopengl>=3.1.4" \
    "pyparsing>=3.0.0" \
    "wrapt>=1.11.2" \
    "pillow" \
    "gymnasium==0.29.1" \
    "PyFlyt==0.4.0" \
    "pygame==2.5.2" \
    "mujoco>=3.6.0" \
    "dm-control" \
    "minigrid==2.3.1" \
    "pandas" \
    "yfinance" \
    "torch" \
    "torchvision" \
    "finrl"

# 6. Configure CARLA
echo ""
echo "Configuring CARLA..."
CARLA_EXE="/Users/studio/Library/Containers/com.isaacmarovitz.Whisky/Bottles/4228CB94-F034-4781-A030-4C610685A4B1/drive_c/Program Files/CARLA_0.9.15/CARLA.app/Contents/SharedSupport/prefix/drive_c/Program Files/CARLA/v9.15/CarlaUE4.exe"

if [ -f "$CARLA_EXE" ]; then
    CARLA_DIR=$(dirname "$CARLA_EXE")
    mkdir -p "$HOME/Applications"
    ln -sfn "$CARLA_DIR" "$HOME/Applications/CARLA_0.9.15"
    ln -sfn "$HOME/Applications/CARLA_0.9.15" "$ROOT_DIR/CARLA_0.9.15"

    # Link Python API
    CARLA_PY=$(find "$(dirname "$CARLA_DIR")" -maxdepth 5 -type d -path "*/PythonAPI/carla" 2>/dev/null | head -1)
    if [ -d "$CARLA_PY" ]; then
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "$CARLA_PY" > "$SITE_PACKAGES/carla.pth"
    fi
    echo "✓ CARLA configured"
else
    echo "⚠ CARLA not found at expected path"
fi

# 7. Build OaK
echo ""
echo "Building OaK..."
swift build -c release

echo ""
echo "✓ Setup complete"
echo ""
echo "To use:"
echo " source $CONDA_DIR/etc/profile.d/conda.sh"
echo " conda activate oak"
echo ".build/release/OaK"