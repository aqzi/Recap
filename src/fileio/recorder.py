import threading

import sounddevice as sd
import soundfile as sf


def list_input_devices() -> list[dict]:
    """List all available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": int(dev["default_samplerate"]),
            })
    return inputs


def record(device_index: int, output_path: str, sample_rate: int = 44100) -> float:
    """Record audio from a device, streaming directly to a WAV file.

    Blocks until the caller signals stop via the returned threading.Event,
    or until the user presses Enter on the main thread.

    Returns the duration in seconds.
    """
    stop_event = threading.Event()
    frames_written = [0]

    with sf.SoundFile(output_path, mode="w", samplerate=sample_rate,
                      channels=1, subtype="PCM_16") as wav:

        def callback(indata, frame_count, time_info, status):
            if stop_event.is_set():
                raise sd.CallbackAbort
            wav.write(indata[:, 0] if indata.shape[1] > 1 else indata)
            frames_written[0] += frame_count

        with sd.InputStream(samplerate=sample_rate, device=device_index,
                            channels=1, dtype="float32", callback=callback,
                            blocksize=4096):
            try:
                input()
            except EOFError:
                pass
            stop_event.set()

    duration = frames_written[0] / sample_rate
    return duration
