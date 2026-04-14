# OaK 2026

Big World compliant agent implementing the Alberta Plan.

## Requirements

- Apple Silicon Mac, current macOS (Tahoe 26+)
- Python 3.11 on your PATH (for example `brew install python@3.11`)
- Xcode 15+ and **Xcode Command Line Tools** (`xcode-select --install`) — needed to build PyBullet from PyPI
- **Homebrew Qt 5** (`brew install qt@5`) — EnvPool’s bundled extensions link against Qt; required for `import envpool` to succeed
- 32GB RAM recommended

## Quick Start

```bash
chmod +x setup.sh
./setup.sh
```

The script prints `PYTHONHOME` as `sys.base_prefix` (where the real stdlib lives), `PYTHONPATH` as the `.venv` `site-packages` directory, and `PYTHON_LIBRARY` for PythonKit. A plain `venv` does not put `encodings` under `.venv`, so **`PYTHONHOME` must not be only `$(pwd)/.venv`**. Copy the four exports from `./setup.sh` output, then:

```bash
swift run -c release -- minigrid
```

## Environments

These run from the same process when `OAK_ENV=all` (or invoke one logical name):

- PyFlyt QuadX (drone hover)
- DM Control Cheetah/Walker/Humanoid
- MiniGrid (navigation)

```
Swift (OaK Core) <--> PythonKit <--> Python Envs
     |
     v
Metal GPU (FIRE, FTL, LZ4)
```
