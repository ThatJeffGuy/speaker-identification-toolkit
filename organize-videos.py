import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn
import multiprocessing

# Initialize Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     VIDEO FILE RENAMING TOOL
=====================================
[/bold cyan]""")
console.print("This tool renames video files by extracting their SxxEyy (Season/Episode) tags.\n")

# Define Directory Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")

# Ensure Directory Exists
if not os.path.exists(VIDEO_DIR):
    console.print(f"[bold red]Error:[/bold red] Directory '{VIDEO_DIR}' not found.")
    exit(1)

# Regex Pattern to Match SXXEYY
pattern = re.compile(r"(S\d{2}E\d{2})", re.IGNORECASE)

# Process Video Files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mkv", ".mp4", ".avi"))]
if not video_files:
    console.print(f"[bold yellow]No video files found in '{VIDEO_DIR}'.[/bold yellow]")
    exit(1)

console.print(f"[bold green]Found {len(video_files)} video file(s). Starting renaming process...[/bold green]")

# Threaded Function to Rename Files
def rename_file(file):
    match = pattern.search(file)
    if match:
        new_name = f"{match.group(1)}{os.path.splitext(file)[1]}"  # Retain file extension
        old_path = os.path.join(VIDEO_DIR, file)
        new_path = os.path.join(VIDEO_DIR, new_name)
        os.rename(old_path, new_path)
        return file, new_name
    return file, None

# Multithreading with Rich Progress Bar
MAX_THREADS = min(8, multiprocessing.cpu_count() - 1)
console.print(f"[bold cyan]Using up to {MAX_THREADS} threads for renaming...[/bold cyan]")

renamed_files = []
with Progress(
    "[progress.percentage]{task.percentage:>3.0f}%",
    BarColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Renaming files...", total=len(video_files))

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(rename_file, file): file for file in video_files}
        for future in as_completed(futures):
            old_file = futures[future]
            try:
                old_name, new_name = future.result()
                if new_name:
                    renamed_files.append((old_name, new_name))
                    console.print(f"[green]Renamed:[/green] {old_name} -> [bold]{new_name}[/bold]")
                else:
                    console.print(f"[yellow]Skipped:[/yellow] {old_file} (no SXXEYY pattern found)")
            except Exception as e:
                console.print(f"[red]Error renaming {old_file}:[/red] {e}")
            progress.update(task, advance=1)

# Final Output
console.print("\n[bold green]Renaming process complete![/bold green]")
if renamed_files:
    console.print(f"[bold cyan]Summary:[/bold cyan]")
    for old, new in renamed_files:
        console.print(f"[white]{old}[/white] -> [bold green]{new}[/bold green]")
else:
    console.print("[yellow]No files were renamed.[/yellow]")

console.print(f"\n[bold green]Check the '{VIDEO_DIR}' directory for results.[/bold green]")

