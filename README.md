# Tactical AI-ANC

Real-time active noise cancellation and speech enhancement using DeepFilterNet, built for low-latency streaming audio.

## Features

- Real-time noise cancellation on microphone input
- Offline batch processing with spectrogram comparison (input vs clean)
- Zero-allocation ring buffer for drop-free audio streaming
- Dual-audio live recording to save both raw and cleaned `.wav` files simultaneously
- Instant export of rolling 10-second history buffer

## Requirements

- Python 3.9+
- `torch`
- `torchaudio`
- `deepfilternet` (`df-enhance`)
- `sounddevice`
- `soundfile`
- `matplotlib`
- `colorama`

### Option 1: Install using `pip`

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchaudio deepfilternet sounddevice soundfile matplotlib colorama
```

### Option 2: Install using `uv` (Faster)

```powershell
# Note: uv must be installed separately first.
# pip install uv

# Create environment and install dependencies:
uv venv venv
uv pip install torch torchaudio deepfilternet sounddevice soundfile matplotlib colorama
```

## How to Run

1. Activate your virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

*(or run the batch launcher)*

2. Start the program:

```bash
python stream_windows.py
```

3. Select your input (microphone) and output (headphones/speakers) from the device list prompt.

## Controls (Real-Time Mode)

- `SPACE`: Toggle AI noise cancellation ON/OFF
- `A`: Start/Stop dual-audio live recording
- `R`: Save rolling 10s buffer (`demo_raw.wav` and `demo_cleaned.wav`)
- `Q` / `ESC`: Exit back to menu

## Acknowledgements

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) by Hendrik Schröter for the neural speech enhancement framework and pre-trained models.
- [PyTorch](https://pytorch.org/) & [torchaudio](https://pytorch.org/audio/) for the tensor computation and neural network runtime.
- [PortAudio](http://www.portaudio.com/) and [python-sounddevice](https://python-sounddevice.readthedocs.io/) for cross-platform low-latency audio I/O streaming.
- [SoundFile](https://python-soundfile.readthedocs.io/) for audio writing and encoding.
- [Matplotlib](https://matplotlib.org/) for spectrogram plotting.
- [Colorama](https://pypi.org/project/colorama/) for terminal styling and metrics display.
- [Astral uv](https://github.com/astral-sh/uv) for fast Python package and environment management.