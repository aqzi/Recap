import os
import sys
from datetime import datetime

from pathvalidate import sanitize_filename

from fileio.progress import console
from fileio.recorder import list_input_devices, record
from rich.prompt import Prompt


def run_recorder(output_dir, record_name):
    """Record audio from an input device and save to a WAV file."""
    console.print()
    console.print("[bold]Audio Recorder[/bold]")
    console.print()

    devices = list_input_devices()
    if not devices:
        console.print("[bold red]Error:[/bold red] No audio input devices found.")
        sys.exit(1)

    console.print("  Available input devices:")
    for dev in devices:
        console.print(f"    [bold cyan][{dev['index']}][/bold cyan] {dev['name']}")
    console.print()

    valid_indices = [str(d["index"]) for d in devices]
    choice = Prompt.ask("Select device", choices=valid_indices)
    device_index = int(choice)

    if output_dir is None:
        output_dir = os.path.join("output", "recordings")
    os.makedirs(output_dir, exist_ok=True)

    if record_name:
        safe_name = sanitize_filename(record_name)
        filename = f"{safe_name}.wav"
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"recording_{stamp}.wav"

    output_path = os.path.join(output_dir, filename)

    console.print()
    console.print(f"  Output: {output_path}")
    console.print()
    console.print("[bold green]Recording... press Enter to stop.[/bold green]")

    duration = record(device_index, output_path)

    minutes = int(duration // 60)
    seconds = int(duration % 60)
    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  File:     {output_path}")
    console.print(f"  Duration: {minutes}m {seconds}s")
