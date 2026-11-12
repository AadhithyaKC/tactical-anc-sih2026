# --- Compatibility shim for newer torchaudio versions ---
import sys, types, torchaudio
if not hasattr(torchaudio, "backend"):
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
    backend.common = common
    torchaudio.backend = backend
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
# --------------------------------------------------------

import os
import time
import msvcrt
import threading
import queue
import sounddevice as sd
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
from df.enhance import enhance, init_df

init(autoreset=True)

# ----------------- FLUSH DENORMALS (DSP OPTIMIZATION) ---
try:
    torch.set_flush_denormal(True)
except Exception:
    pass

# ----------------- GLOBAL CONFIGURATION -----------------
SAMPLE_RATE = 48000
BLOCK_SIZE = 960           # 20ms frame buffer at 48kHz
BUFFER_CONTEXT_LEN = 2880  # 60ms rolling context
RAW_GAIN = 1.0
AI_GAIN = 2.5
AI_CEILING = 0.98
DURATION = 5
HISTORY_TOTAL_SAMPLES = 10 * SAMPLE_RATE

ui_queue = queue.Queue(maxsize=5)
disk_rec_queue = queue.Queue(maxsize=500)

# ----------------- FAST PRE-ALLOCATED AUDIO RING BUFFER -
class AudioBlockRing:
    """Pre-allocated ring buffer for passing fixed-size audio blocks between threads."""
    def __init__(self, capacity=4, block_len=BLOCK_SIZE):
        self.buffer = np.zeros((capacity, block_len), dtype=np.float32)
        self.capacity = capacity
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.buffer[self.write_pos][:] = data
            self.write_pos = (self.write_pos + 1) % self.capacity
            if self.count < self.capacity:
                self.count += 1
            else:
                self.read_pos = (self.read_pos + 1) % self.capacity  # drop oldest on overrun

    def read(self, out):
        with self.lock:
            if self.count == 0:
                return False
            out[:] = self.buffer[self.read_pos]
            self.read_pos = (self.read_pos + 1) % self.capacity
            self.count -= 1
            return True

# ----------------- DEVICE SELECTION HELPER --------------
def configure_devices():
    devices = sd.query_devices()

    print("\n" + Fore.CYAN + "=" * 65)
    print(Fore.WHITE + Style.BRIGHT + "  AVAILABLE AUDIO HARDWARE ON THIS SYSTEM")
    print(Fore.CYAN + "=" * 65)

    inputs, outputs = [], []
    for idx, dev in enumerate(devices):
        max_in = dev.get('max_input_channels', 0)
        max_out = dev.get('max_output_channels', 0)
        api_name = sd.query_hostapis(dev['hostapi'])['name']

        if max_in > 0:
            inputs.append((idx, dev['name'], api_name, max_in))
        if max_out > 0:
            outputs.append((idx, dev['name'], api_name, max_out))

    print(Fore.YELLOW + "\n--- INPUT DEVICES (Microphones) ---")
    for idx, name, api, ch in inputs:
        print(f"  [{idx:2d}] {name} ({api}, {ch} in)")

    print(Fore.YELLOW + "\n--- OUTPUT DEVICES (3.5mm / Speakers) ---")
    for idx, name, api, ch in outputs:
        print(f"  [{idx:2d}] {name} ({api}, {ch} out)")

    print(Fore.CYAN + "=" * 65)

    while True:
        try:
            in_choice = input(Fore.WHITE + "Enter INPUT device index: ").strip()
            in_idx = int(in_choice)
            if any(d[0] == in_idx for d in inputs):
                break
            print(Fore.RED + "Invalid index.")
        except ValueError:
            print(Fore.RED + "Numeric index required.")

    while True:
        try:
            out_choice = input(Fore.WHITE + "Enter OUTPUT device index: ").strip()
            out_idx = int(out_choice)
            if any(d[0] == out_idx for d in outputs):
                break
            print(Fore.RED + "Invalid index.")
        except ValueError:
            print(Fore.RED + "Numeric index required.")

    # Default to mono (1 ch) to avoid halving signal gain on stereo interfaces with 1 mic
    in_channels = 1
    sd.default.device = (in_idx, out_idx)
    print(Fore.GREEN + f"\n[ROUTING OK] Mic ID: {in_idx} ({in_channels} ch) -> Output ID: {out_idx} (2 ch)\n")
    return in_idx, out_idx, in_channels

INPUT_DEVICE_ID, OUTPUT_DEVICE_ID, INPUT_CHANNELS = configure_devices()

# ----------------- INITIALIZATION -----------------------
print(Fore.CYAN + "[INIT] Initializing Neural Inference Engine...")
torch.set_num_threads(min(4, max(2, (os.cpu_count() or 4) // 2)))
model, df_state, _ = init_df()

if hasattr(df_state, "set_atten_lim"):
    df_state.set_atten_lim(28.0)

# Pre-allocated memory tensors and DSP scratch buffers
rolling_raw_buffer = np.zeros(BUFFER_CONTEXT_LEN, dtype=np.float32)
rolling_tensor = torch.from_numpy(rolling_raw_buffer).unsqueeze(0)

raw_scratch = np.zeros(BLOCK_SIZE, dtype=np.float32)
ai_scratch = np.zeros(BLOCK_SIZE, dtype=np.float32)
proc_audio_buf = np.zeros(BLOCK_SIZE, dtype=np.float32)
clean_record_buf = np.zeros(BLOCK_SIZE, dtype=np.float32)

# Pre-allocated 10-second rolling history circular buffers
raw_history_buf = np.zeros(HISTORY_TOTAL_SAMPLES, dtype=np.float32)
clean_history_buf = np.zeros(HISTORY_TOTAL_SAMPLES, dtype=np.float32)
history_write_idx = 0
history_lock = threading.Lock()

# ----------------- ASYNC DISK RECORDER WORKER -----------
def disk_writer_worker():
    raw_writer, clean_writer = None, None
    while True:
        item = disk_rec_queue.get()
        if item is None:
            break
        cmd, payload = item
        if cmd == "START":
            raw_path, clean_path = payload
            raw_writer = sf.SoundFile(raw_path, mode='w', samplerate=SAMPLE_RATE, channels=1)
            clean_writer = sf.SoundFile(clean_path, mode='w', samplerate=SAMPLE_RATE, channels=1)
        elif cmd == "WRITE":
            raw_blk, clean_blk = payload
            if raw_writer:
                raw_writer.write(raw_blk)
            if clean_writer:
                clean_writer.write(clean_blk)
        elif cmd == "STOP":
            if raw_writer:
                raw_writer.close()
                raw_writer = None
            if clean_writer:
                clean_writer.close()
                clean_writer = None
        disk_rec_queue.task_done()

threading.Thread(target=disk_writer_worker, daemon=True).start()

# ----------------- AUDIO CUE GENERATOR ------------------
def make_beep(freq, duration=0.06):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    fade = int(0.01 * SAMPLE_RATE)
    envelope = np.ones_like(t)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (0.12 * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)

BEEP_AI_ON = make_beep(880, 0.06)
BEEP_AI_OFF = make_beep(350, 0.09)
BEEP_REC_START = make_beep(1200, 0.08)
BEEP_REC_STOP = make_beep(600, 0.08)
cue_buffer = np.array([], dtype=np.float32)
cue_lock = threading.Lock()

def trigger_cue(beep_arr):
    global cue_buffer
    with cue_lock:
        cue_buffer = beep_arr.copy()

# ----------------- STREAM RUNTIME STATE -----------------
anc_enabled = False
is_recording_live = False

def get_next_capture_index():
    idx = 1
    while os.path.exists(f"captured_{idx}_raw.wav") or os.path.exists(f"captured_{idx}_cleaned.wav"):
        idx += 1
    return idx

def toggle_anc():
    global anc_enabled
    anc_enabled = not anc_enabled
    trigger_cue(BEEP_AI_ON if anc_enabled else BEEP_AI_OFF)

def toggle_live_recording():
    global is_recording_live
    if not is_recording_live:
        idx = get_next_capture_index()
        raw_file = f"captured_{idx}_raw.wav"
        clean_file = f"captured_{idx}_cleaned.wav"
        disk_rec_queue.put(("START", (raw_file, clean_file)))
        is_recording_live = True
        trigger_cue(BEEP_REC_START)
        print(Fore.RED + f"\n[LIVE REC START] Saving to '{raw_file}' & '{clean_file}'...")
    else:
        is_recording_live = False
        trigger_cue(BEEP_REC_STOP)
        disk_rec_queue.put(("STOP", None))
        print(Fore.GREEN + "\n[LIVE REC STOPPED] Files flushed and finalized.")

def save_buffer_recording():
    with history_lock:
        idx = history_write_idx
        raw_audio = np.concatenate([raw_history_buf[idx:], raw_history_buf[:idx]])
        clean_audio = np.concatenate([clean_history_buf[idx:], clean_history_buf[:idx]])
    sf.write("demo_raw.wav", raw_audio, SAMPLE_RATE)
    sf.write("demo_cleaned.wav", clean_audio, SAMPLE_RATE)
    print(Fore.YELLOW + "\n[BUFFER EXPORT] Saved rolling 10s buffer to 'demo_raw.wav' & 'demo_cleaned.wav'")

# ----------------- MODE 1: ULTRA-LOW LATENCY STREAM -----
def run_live_stream():
    global is_recording_live, history_write_idx
    print(Fore.CYAN + "\n[WARMUP] Binding low-latency audio hardware streams...")

    in_ring = AudioBlockRing(capacity=4, block_len=BLOCK_SIZE)
    out_ring = AudioBlockRing(capacity=4, block_len=BLOCK_SIZE)
    in_transfer_buf = np.zeros(BLOCK_SIZE, dtype=np.float32)
    hw_out_buf = np.zeros(BLOCK_SIZE, dtype=np.float32)
    stop_event = threading.Event()

    def inference_worker():
        global history_write_idx
        frame_count = 0
        latency_accumulator = 0.0

        while not stop_event.is_set():
            if not in_ring.read(raw_scratch):
                time.sleep(0.001)
                continue

            t_start = time.perf_counter()

            # In-place shift of context buffer
            rolling_raw_buffer[:-BLOCK_SIZE] = rolling_raw_buffer[BLOCK_SIZE:]
            rolling_raw_buffer[-BLOCK_SIZE:] = raw_scratch

            # Inference branch (decoupled from PortAudio hardware callback)
            if anc_enabled or is_recording_live:
                with torch.inference_mode():
                    cleaned_tensor = enhance(model, df_state, rolling_tensor, pad=False)

                ai_scratch[:] = cleaned_tensor[0, -BLOCK_SIZE:].cpu().numpy()

                np.multiply(ai_scratch, AI_GAIN, out=clean_record_buf)
                np.tanh(clean_record_buf, out=clean_record_buf)
                np.clip(clean_record_buf, -AI_CEILING, AI_CEILING, out=clean_record_buf)

                if anc_enabled:
                    # 90% AI speech + 10% passthrough blend in-place
                    np.multiply(ai_scratch, 0.90, out=proc_audio_buf)
                    proc_audio_buf[:] += raw_scratch * 0.10
                    proc_audio_buf[:] *= AI_GAIN
                    np.tanh(proc_audio_buf, out=proc_audio_buf)
                    np.clip(proc_audio_buf, -AI_CEILING, AI_CEILING, out=proc_audio_buf)
                else:
                    np.multiply(raw_scratch, RAW_GAIN, out=proc_audio_buf)
                    np.clip(proc_audio_buf, -AI_CEILING, AI_CEILING, out=proc_audio_buf)
            else:
                # 100% Zero-Inference bypass
                np.multiply(raw_scratch, RAW_GAIN, out=proc_audio_buf)
                np.clip(proc_audio_buf, -AI_CEILING, AI_CEILING, out=proc_audio_buf)
                clean_record_buf[:] = proc_audio_buf

            # Push processed block to playback ring immediately
            out_ring.write(proc_audio_buf)

            # Update rolling 10s history safely
            with history_lock:
                raw_history_buf[history_write_idx:history_write_idx + BLOCK_SIZE] = raw_scratch
                clean_history_buf[history_write_idx:history_write_idx + BLOCK_SIZE] = proc_audio_buf
                history_write_idx = (history_write_idx + BLOCK_SIZE) % HISTORY_TOTAL_SAMPLES

            if is_recording_live:
                try:
                    disk_rec_queue.put_nowait(("WRITE", (raw_scratch.copy(), clean_record_buf.copy())))
                except queue.Full:
                    pass

            t_end = time.perf_counter()
            latency_accumulator += (t_end - t_start) * 1000.0
            frame_count += 1

            if frame_count % 10 == 0:
                avg_ms = latency_accumulator / 10.0
                in_pk = float(np.max(np.abs(raw_scratch)))
                out_pk = float(np.max(np.abs(proc_audio_buf)))

                try:
                    ui_queue.put_nowait((anc_enabled, in_pk, out_pk, avg_ms, is_recording_live))
                except queue.Full:
                    pass
                latency_accumulator = 0.0

    def input_callback(indata, frames, time_info, status):
        if indata.shape[1] > 1:
            in_transfer_buf[:] = indata[:, 0]
        else:
            in_transfer_buf[:] = indata[:, 0]
        in_ring.write(in_transfer_buf)

    def output_callback(outdata, frames, time_info, status):
        global cue_buffer
        if not out_ring.read(hw_out_buf):
            outdata.fill(0)
            return

        # Mix audio cue if active
        with cue_lock:
            if len(cue_buffer) > 0:
                chunk_len = min(BLOCK_SIZE, len(cue_buffer))
                hw_out_buf[:chunk_len] += cue_buffer[:chunk_len]
                cue_buffer = cue_buffer[chunk_len:]

        outdata[:, 0] = hw_out_buf
        if outdata.shape[1] > 1:
            outdata[:, 1] = hw_out_buf

    worker_thread = threading.Thread(target=inference_worker, daemon=True)
    worker_thread.start()

    in_stream = sd.InputStream(
        device=INPUT_DEVICE_ID,
        channels=INPUT_CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        latency='low',
        callback=input_callback
    )

    out_stream = sd.OutputStream(
        device=OUTPUT_DEVICE_ID,
        channels=2,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        latency='low',
        callback=output_callback
    )

    print("\n" + Fore.GREEN + "=" * 78)
    print(Fore.WHITE + Style.BRIGHT + "  NOISEFILTER_AI LOW-LATENCY MONITOR (<20ms FRAME BUDGET)")
    print(Fore.WHITE + "  [SPACE] Toggle AI/RAW  |  [A] Toggle Live Dual-Audio Recording")
    print(Fore.WHITE + "  [R] Export Last 10s    |  [Q/ESC] Return to Menu")
    print(Fore.GREEN + "=" * 78 + "\n")

    try:
        with in_stream, out_stream:
            while True:
                try:
                    enabled, in_pk, out_pk, avg_ms, rec_active = ui_queue.get_nowait()
                    reduction_db = (
                        max(0.0, 20.0 * np.log10((in_pk + 1e-5) / (out_pk + 1e-5)))
                        if in_pk > 1e-4 and out_pk > 1e-4 else 0.0
                    )
                    budget_str = (
                        Fore.GREEN + f"{avg_ms:4.1f}ms [PASS]"
                        if avg_ms < 20.0 else Fore.RED + f"{avg_ms:4.1f}ms [FAIL]"
                    )
                    mode_str = Fore.GREEN + "[NOISEFILTER_AI ON]" if enabled else Fore.YELLOW + "[NOISEFILTER_AI OFF]"
                    rec_str = Fore.RED + "[● REC]" if rec_active else Fore.WHITE + "[IDLE]"

                    raw_bar = "|" * int(min(1.0, in_pk * 3.5) * 10)
                    out_bar = "|" * int(min(1.0, out_pk * 3.5) * 10)
                    raw_bar += " " * (10 - len(raw_bar))
                    out_bar += " " * (10 - len(out_bar))

                    dash = (
                        f"\r{rec_str} {mode_str} INPUT:[{raw_bar}] OUTPUT:[{out_bar}] "
                        f"| -{reduction_db:4.1f}dB | Inference: {budget_str}   "
                    )
                    sys.stdout.write(dash)
                    sys.stdout.flush()
                except queue.Empty:
                    pass

                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':
                        toggle_anc()
                    elif key in (b'a', b'A'):
                        toggle_live_recording()
                    elif key in (b'r', b'R'):
                        save_buffer_recording()
                    elif key in (b'q', b'Q', b'\x1b', b'\r'):
                        if is_recording_live:
                            toggle_live_recording()
                        print(Fore.YELLOW + "\n[INFO] Returning to Master Menu...")
                        break
                time.sleep(0.01)
    finally:
        stop_event.set()
        worker_thread.join(timeout=1.0)

# ----------------- MODE 2: BATCH & SPECTROGRAM ----------
def run_batch_demo():
    print("\n" + Fore.CYAN + "=" * 65)
    print(Fore.WHITE + Style.BRIGHT + "  NOISEFILTER_AI BATCH BENCHMARK & SPECTROGRAM")
    print(Fore.CYAN + "=" * 65)
    input(Fore.WHITE + ">>> Press [ENTER] to record 5 seconds... <<<")

    print(Fore.RED + f"\n[*] RECORDING {DURATION} SECONDS...")
    raw_audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        device=INPUT_DEVICE_ID,
        samplerate=SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        dtype='float32'
    )
    sd.wait()

    raw_channel = raw_audio[:, 0] if raw_audio.ndim > 1 else raw_audio.reshape(-1)
    raw_audio_1d = np.ascontiguousarray(raw_channel, dtype=np.float32)
    sf.write("batch_input.wav", raw_audio_1d, SAMPLE_RATE)

    print(Fore.CYAN + "[*] PROCESSING: DeepFilterNet STFT Analysis...")
    audio_tensor = torch.from_numpy(raw_audio_1d).unsqueeze(0).contiguous()

    # Instantiate a fresh DF state without reloading weights from disk
    try:
        batch_df_state = type(df_state)(
            sr=df_state.sr(),
            fft_size=df_state.fft_size(),
            hop_size=df_state.hop_size(),
            nb_bands=df_state.nb_bands(),
        )
    except Exception:
        _, batch_df_state, _ = init_df()

    if hasattr(batch_df_state, "set_atten_lim"):
        batch_df_state.set_atten_lim(28.0)

    with torch.inference_mode():
        clean_tensor = enhance(model, batch_df_state, audio_tensor)

    clean_audio_1d = np.ascontiguousarray(clean_tensor.squeeze().cpu().numpy(), dtype=np.float32)
    sf.write("batch_cleaned.wav", clean_audio_1d, SAMPLE_RATE)
    print(Fore.GREEN + "[SUCCESS] Clean audio generated!\n")

    raw_stereo = np.column_stack([raw_audio_1d, raw_audio_1d])
    clean_stereo = np.column_stack([clean_audio_1d, clean_audio_1d])

    while True:
        print(Fore.WHITE + "-" * 55)
        print("DEMO CONTROLS: [1] Raw | [2] Clean | [3] Spectrogram | [S] Stop | [B] Back")
        choice = input("Select [1/2/3/S/B]: ").strip().lower()

        if choice == '1':
            sd.stop()
            sd.play(raw_stereo, samplerate=SAMPLE_RATE, device=OUTPUT_DEVICE_ID)
        elif choice == '2':
            sd.stop()
            sd.play(clean_stereo, samplerate=SAMPLE_RATE, device=OUTPUT_DEVICE_ID)
        elif choice == 's':
            sd.stop()
        elif choice == '3':
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            axes[0].specgram(raw_audio_1d, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap='inferno')
            axes[0].set_title("Input (Corrupted)", fontsize=11, fontweight='bold')
            axes[0].set_ylabel("Frequency (Hz)")

            axes[1].specgram(clean_audio_1d, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap='inferno')
            axes[1].set_title("Output (AI Cleaned)", fontsize=11, fontweight='bold')
            axes[1].set_ylabel("Frequency (Hz)")
            axes[1].set_xlabel("Time (seconds)")
            plt.tight_layout()
            plt.show()
        elif choice == 'b':
            sd.stop()
            break

# ----------------- MASTER ENTRY POINT -------------------
def main():
    while True:
        print("\n" + Fore.GREEN + "==================================================")
        print(Fore.WHITE + Style.BRIGHT + "    NOISEFILTER_AI DEFENSE AI-ANC CONTROL SUITE")
        print(Fore.GREEN + "==================================================")
        print("  [1] Real Time Stream (For Live Communications)")
        print("  [2] Superior Audio Processing (For General Communications)")
        print("  [Q] Exit")
        choice = input("\nSelect Mode [1/2/Q]: ").strip().lower()

        if choice == '1':
            run_live_stream()
        elif choice == '2':
            run_batch_demo()
        elif choice in ('q', 'exit'):
            break

if __name__ == "__main__":
    main()
