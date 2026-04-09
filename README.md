# OaK 2026

Big World compliant agent implementing the Alberta Plan.

## Requirements

- macOS 15+ (M1/M2/M3/M4/M5)
- Python 3.11
- Xcode 15+
- 32GB RAM recommended
- CARLA 0.9.15 server

## Quick Start

```bash
# 1. Setup
chmod +x setup.sh
./setup.sh

# 2. Start CARLA (in separate terminal)
./CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen

# 3. Run
source venv/bin/activate
.build/release/OaK
```


## Environments
### All 8 run concurrently in same process:


- PyFlyt QuadX (drone hover)
- CARLA (autonomous driving)
- DM Control Cheetah/Walker/Humanoid
- MiniGrid (navigation)
- Habitat (embodied AI)
- FinRL (trading)


Swift (OaK Core) <--> PythonKit <--> Python Envs
     |
     v
Metal GPU (FIRE, FTL, LZ4)



# For Carla on mac os we need to do this

## After running setup.sh you will see these steps

```
🍺  whisky was successfully installed!

MANUAL STEP REQUIRED:
1. Download the pre-packaged CARLA for Mac (21GB):
 https://github.com/carla-simulator/carla/discussions/9037
 Look for 'You can download the package here' in the discussion

2. Open Whisky, create new bottle named 'CARLA'
3. Install Windows Python 3.10 in the bottle
4. Extract CARLA to ~/Applications/CARLA_0.9.15

See PDF: CARLA Server on Apple Silicon Mac.pdf
in the discussion for full steps
```

### I chose the windows 10 option