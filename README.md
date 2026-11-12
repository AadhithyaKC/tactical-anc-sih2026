# Tactical AI-ANC

Real-time AI noise suppression and speech enhancement using DeepFilterNet, built for low-latency uplink audio streaming[cite: 3].

## Features

- Real-time neural noise suppression on microphone input[cite: 3]
- Decoupled worker-thread inference with zero-allocation ring buffers for drop-free audio I/O[cite: 3, 4]
- Offline batch processing with input vs. cleaned spectrogram comparison[cite: 3, 4]
- Dual-audio live recording to capture raw and cleaned `.wav` files simultaneously[cite: 3, 4]
- Instant export of a rolling 10-second audio history buffer[cite: 3, 4]

## Requirements

- Windows 10/11 (uses `msvcrt` for non-blocking terminal hotkeys)
- Python 3.9+[cite: 3]
- Dependencies listed in `requirements.txt` (`torch`, `torchaudio`, `deepfilternet`, `numpy`, `sounddevice`, `soundfile`, `matplotlib`, `colorama`)[cite: 6]

### Option 1: Install using `pip`

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 2: Install using `uv` (Faster)

```powershell
# Note: uv must be installed separately first (pip install uv)
uv venv venv
uv pip install -r requirements.txt
```

## How to Run

### Quick Launch (Windows)

Double-click `run.bat` (or `launch.bat`) to activate the virtual environment and start the application automatically.

### Manual Launch

1. Activate your virtual environment[cite: 3]:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

2. Start the program[cite: 3]:

```bash
python stream_windows.py
```

3. Select your input (microphone) and output (headphones/speakers) device indices from the hardware prompt[cite: 3, 4].

## Controls (Real-Time Mode)

- `SPACE`: Toggle AI noise suppression ON/OFF[cite: 3, 4]
- `A`: Start/Stop dual-audio live recording (`captured_<n>_raw.wav` & `captured_<n>_cleaned.wav`)[cite: 3, 4]
- `R`: Export rolling 10-second buffer (`demo_raw.wav` and `demo_cleaned.wav`)[cite: 3, 4]
- `Q` / `ESC`: Exit back to main menu[cite: 3, 4]

## Troubleshooting

If no audio is detected or you want to verify microphone input levels outside the main app, run this quick loopback check with your virtual environment activated[cite: 2, 3]:

```powershell
python -c "import sounddevice as sd, numpy as np; print('Speaking test active (Ctrl+C to quit)...'); sd.Stream(channels=1, samplerate=48000, blocksize=960, callback=lambda ind, outd, f, t, s: (outd.__setitem__(slice(None), ind * 2.0), print(f'Mic Level: {np.abs(ind).mean():.4f}', end='\r'))).start(); input()"
```

## License

Licensed under the [MIT License](LICENSE.md)[cite: 8].

## Acknowledgements

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) by Hendrik Schröter for the neural speech enhancement framework and pre-trained models[cite: 3].
- [PyTorch](https://pytorch.org/) & [torchaudio](https://pytorch.org/audio/) for tensor computation and neural network runtime[cite: 3].
- [PortAudio](http://www.portaudio.com/) and [python-sounddevice](https://python-sounddevice.readthedocs.io/) for low-latency audio I/O streaming[cite: 3].
- [SoundFile](https://python-soundfile.readthedocs.io/) for audio encoding and disk I/O[cite: 3].
- [Matplotlib](https://matplotlib.org/) for spectrogram visualization[cite: 3].
- [Colorama](https://pypi.org/project/colorama/) for terminal HUD styling[cite: 3].
- [Astral uv](https://github.com/astral-sh/uv) for fast Python environment and package management[cite: 3].
