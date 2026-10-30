# Tactical AI-ANC: Neural Active Noise Cancellation & Speech Enhancement

[![Smart India Hackathon](https://img.shields.io/badge/SIH-2026-orange.svg)](https://sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/Problem%20Statement-SIH26052-blue.svg)](https://sih.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time, low-latency neural active noise cancellation (ANC) and speech enhancement system built on **DeepFilterNet** and designed for tactical communications, field operations, and edge streaming.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Mechanics](#architecture--mechanics)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
  - [Prerequisites](#prerequisites)
  - [Option A: Standard pip Installation](#option-a-standard-pip-installation)
  - [Option B: Fast uv Installation](#option-b-fast-uv-installation)
- [Running the Application](#running-the-application)
  - [Interactive Real-Time Mode](#interactive-real-time-mode)
  - [Batch / Offline Mode](#batch--offline-mode)
  - [Using the Windows One-Click Launcher](#using-the-windows-one-click-launcher)
- [Live Controls & Keybindings](#live-controls--keybindings)
- [Directory Structure](#directory-structure)
- [Troubleshooting & Audio Setup](#troubleshooting--audio-setup)
- [Third-Party Acknowledgements](#third-party-acknowledgements)
- [License](#license)

---

## Overview

High-noise environments (tactical gear, combat vehicles, industrial machinery, and field headsets) degrade verbal comprehension and overload traditional spectral-subtraction filters.

**Tactical AI-ANC** leverages deep filtering algorithms to isolate speech formants and suppress non-stationary acoustic noise in sub-10ms frame budgets. It incorporates a zero-allocation ring buffer, continuous dual-stream recording (raw vs. enhanced), and spectral diagnostic tools.

---

## Key Features

- **Real-Time Stream Enhancement**: Ultra-low-latency neural speech filtering on live microphone feeds.
- **Zero-Allocation Ring Buffer**: Eliminates memory fragmentation and audio frame drops during extended streaming sessions.
- **Dual-Audio Live Recording**: Concurrently exports separate uncompressed `.wav` recordings for unprocessed mic input and denoised output.
- **Rolling History Buffer**: Instant one-key extraction of the most recent 10 seconds of streaming audio (`demo_raw.wav` and `demo_cleaned.wav`).
- **Offline Batch Processing**: Enhances pre-recorded audio files with automatic side-by-side spectrogram generation for signal inspection.
- **Cross-Platform Compatibility**: Tested on Windows 10/11 with support for standard PortAudio-compatible hardware.

---

## Architecture & Mechanics

```
 [Microphone Input] 
         │
         ▼
 [PortAudio Stream Callback]
         │
         ▼
 [Zero-Allocation Ring Buffer]
         │
         ▼
 [DeepFilterNet Inference Engine] ◄── [Trained Weights / PyTorch Runtime]
         │
         ├──► [Real-Time Audio Output / Headphones]
         │
         └──► [Dual Storage Engine] ──► (Rolling Buffer / Live Capture .wav)
```

1. **Audio Ingestion**: Captured at 48 kHz via `sounddevice` and buffered into frame chunks matching DeepFilterNet's internal hop size.
2. **Filtering**: Two-stage deep filtering using ERB (Equivalent Rectangular Bandwidth) scale gain calculation followed by complex spectral filtering.
3. **Latency Control**: Bypasses operating-system mixing layers where possible to deliver real-time feedback with minimal phase distortion.

---

## System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS.
- **Python**: Version 3.9, 3.10, or 3.11.
- **Hardware**:
  - Dedicated microphone or communications headset.
  - Headphones (strongly recommended to avoid acoustic feedback loops during live output monitoring).
  - Optional: CUDA-capable NVIDIA GPU for accelerated batched inferencing.

---

## Installation Guide

### Prerequisites

Clone the repository and enter the project folder:

```bash
git clone https://github.com/AadhithyaKC/tactical-anc-sih2026.git
cd tactical-anc-sih2026
```

---

### Option A: Standard `pip` Installation

1. **Create and activate a virtual environment:**

   ```powershell
   python -m venv venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**

   - **For Standard / CPU Systems:**
     ```powershell
     pip install torch torchaudio deepfilternet sounddevice soundfile matplotlib colorama
     ```

   - **For NVIDIA GPU Systems (CUDA 12.1):**
     ```powershell
     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install deepfilternet sounddevice soundfile matplotlib colorama
     ```

---

### Option B: Fast `uv` Installation

[`uv`](https://github.com/astral-sh/uv) installs packages significantly faster through parallel downloads and global wheel caching.

1. **Install `uv` (if not already installed):**

   ```powershell
   pip install uv
   # Or via official installer:
   # powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Create the environment and install:**

   ```powershell
   uv venv venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
   uv pip install torch torchaudio deepfilternet sounddevice soundfile matplotlib colorama
   ```

---

## Running the Application

### Interactive Real-Time Mode

1. Activate your virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Start the processing engine:
   ```powershell
   python stream_windows.py
   ```

3. Select your hardware devices:
   - Input: Choose the index corresponding to your microphone.
   - Output: Choose the index corresponding to your headphones or monitoring speakers.

---

### Batch / Offline Mode

To clean pre-recorded `.wav` files without microphone streaming, pass the input path directly:

```powershell
python -m df.enhance --output-dir ./enhanced_audio path/to/noisy_recording.wav
```

This generates:
- Cleaned speech audio (`.wav`)
- Spectrogram visual comparison showing noise attenuation across frequency bands

---

### Using the Windows One-Click Launcher

For quick deployment on field machines, double-click `Launch_ANC.bat` from the root directory. This batch script automatically activates the local `venv/` and starts `stream_windows.py`.

---

## Live Controls & Keybindings

While real-time mode is running in the terminal:

| Key | Function | Description |
|:---:|---|---|
| <kbd>SPACE</kbd> | **Toggle ANC** | Switch between raw pass-through and neural noise cancellation |
| <kbd>A</kbd> | **Dual Record** | Start/stop concurrent recording of raw and cleaned audio streams |
| <kbd>R</kbd> | **Dump Rolling Buffer** | Immediately save the last 10 seconds of raw and clean audio to `.wav` |
| <kbd>Q</kbd> / <kbd>ESC</kbd> | **Exit** | Gracefully terminate audio streams, flush buffers, and exit |

---

## Directory Structure

```text
tactical-anc-sih2026/
├── .gitignore               # Ignores venv/, build artifacts, and saved audio
├── Launch_ANC.bat           # Windows quick-launch executable script
├── README.md                # Project documentation and specifications
├── stream_windows.py        # Core real-time stream loop, ring buffer, & UI
├── E drive access.txt       # Local environment notes & path reference
└── Mic.txt                  # Audio device configurations & interface indices
```

---

## Troubleshooting & Audio Setup

- **Acoustic Feedback / Howling**: If you hear loud screeching when starting the engine, your microphone is picking up the output from your speakers. **Switch to headphones** or select a different output device.
- **PortAudio Underflow / Stuttering**:
  - Close other background applications utilizing high CPU or exclusive audio locks (e.g., Discord, ASIO DAW sessions).
  - Increase the buffer block size within `stream_windows.py` if running on slower dual-core hardware.
- **PowerShell Script Execution Error**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating `venv`.

---

## Third-Party Acknowledgements

- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** (Hendrik Schröter): Full neural speech enhancement architecture, model checkpoints, and deep filtering engine.
- **[PyTorch](https://pytorch.org/) & [torchaudio](https://pytorch.org/audio/)**: Tensor compute backend and audio preprocessing utilities.
- **[PortAudio](http://www.portaudio.com/) & [python-sounddevice](https://python-sounddevice.readthedocs.io/)**: Low-latency, multi-platform streaming audio I/O bindings.
- **[SoundFile](https://python-soundfile.readthedocs.io/)**: Clean 24-bit/16-bit PCM `.wav` export and serialization.
- **[Matplotlib](https://matplotlib.org/)**: Spectral visualization and signal analysis utilities.
- **[Colorama](https://pypi.org/project/colorama/)**: ANSI escape character support for the live terminal status UI.
- **[Astral uv](https://github.com/astral-sh/uv)**: Next-generation fast packaging and environment tooling.

---

## License

This project is released under the [MIT License](LICENSE).