# ANC Simulator - Laptop Version

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Simulation
```bash
python anc_simulator.py --duration 10 --noise-type mix --save-plot output.png
```

## Usage Examples

### Stationary Noise Only
```bash
python anc_simulator.py --duration 5 --noise-type stationary --save-plot stationary.png
```

### Non-Stationary (Chirp) Noise
```bash
python anc_simulator.py --duration 5 --noise-type nonstationary --save-plot nonstationary.png
```

### Impulsive Noise
```bash
python anc_simulator.py --duration 5 --noise-type impulsive --save-plot impulsive.png
```

### Mixed Noise (Default)
```bash
python anc_simulator.py --duration 10 --noise-type mix --save-plot mixed.png
```

### With Real-Time Display
```bash
python anc_simulator.py --duration 10 --enable-display --save-plot output.png
```

### Custom Parameters
```bash
python anc_simulator.py \
  --duration 15 \
  --samplerate 16000 \
  --filter-len 64 \
  --block-ms 20 \
  --noise-type mix \
  --save-plot custom_output.png
```

## Parameters

- `--duration`: Simulation duration in seconds (default: 10)
- `--samplerate`: Sample rate in Hz (default: 16000)
- `--filter-len`: Adaptive filter length (default: 32)
- `--block-ms`: Processing block size in ms (default: 20)
- `--noise-type`: Type of noise - stationary/nonstationary/impulsive/mix (default: mix)
- `--enable-display`: Show real-time metrics
- `--save-plot`: Output plot filename (default: anc_output.png)

## Output

The simulator generates:
1. **Console output** with ANC performance metrics
2. **PNG plot** with 3 subplots:
   - Input Signals (Reference + Measured Error)
   - Generated Anti-Noise
   - Residual Error After ANC

## Features

✅ Multi-algorithm ANC (FxLMS, VSSLMS, Robust)
✅ AI/ML-based noise classification
✅ Real-time signal processing simulation
✅ Noise reduction metrics (dB)
✅ Visualizations
✅ No GPIO required (laptop friendly)

## System Requirements

- Python 3.7+
- Windows/Mac/Linux
- 4GB RAM minimum
- ~2 seconds per second of simulated audio
