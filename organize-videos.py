import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn
import multiprocessing
import signal
import sys
import atexit

# Initialize Console
console = Console()
console.clear()

# Global flag for exit request
exit_requested = False

# Signal handler for graceful exit
def signal_handler(sig, frame):
    global exit_requested
    exit_requested = True
    console.print("\n[bold yellow]Interrupt received. Finishing current tasks and exiting...[/bold yellow]")
    # Don't exit immediately, let the program exit cleanly

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register exit function to ensure it runs no matter how the script exits
def exit_handler():
    print("\n")  # Add a blank line for spacing
    console.print("[bold cyan]Press any key to exit...[/bold cyan]")
    
    # Use msvcrt on Windows to capture any key without requiring Enter
    try:
        import msvcrt
        msvcrt.getch()
    except ImportError:
        # Fallback for non-Windows platforms
        input()

atexit.register(exit_handler)

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
    sys.exit(1)

# Regex Pattern to Match SXXEYY
pattern = re.compile(r"(S\d{2}E\d{2})", re.IGNORECASE)

# Process Video Files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mkv", ".mp4", ".avi"))]
if not video_files:
    console.print(f"[bold yellow]No video files found in '{VIDEO_DIR}'.[/bold yellow]")
    sys.exit(1)

console.print(f"[bold green]Found {len(video_files)} video file(s). Starting renaming process...[/bold green]")

# Threaded Function to Rename Files
def rename_file(file):
    global exit_requested
    if exit_requested:
        return file, None, "Skipped due to exit request"
    
    try:
        match = pattern.search(file)
        if match:
            new_name = f"{match.group(1)}{os.path.splitext(file)[1]}"  # Retain file extension
            old_path = os.path.join(VIDEO_DIR, file)
            new_path = os.path.join(VIDEO_DIR, new_name)
            os.rename(old_path, new_path)
            return file, new_name, "Success"
        return file, None, "No pattern match"
    except Exception as e:
        return file, None, f"Error: {str(e)}"

# Multithreading with Rich Progress Bar
MAX_THREADS = min(8, multiprocessing.cpu_count() - 1)
console.print(f"[bold cyan]Using up to {MAX_THREADS} threads for renaming...[/bold cyan]")

renamed_files = []
skipped_files = []
error_files = []

try:
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
                    old_name, new_name, status = future.result()
                    
                    if status == "Success":
                        renamed_files.append((old_name, new_name))
                        progress.console.log(f"[green]Renamed:[/green] {old_name} -> [bold]{new_name}[/bold]")
                    elif status == "No pattern match":
                        skipped_files.append(old_name)
                        progress.console.log(f"[yellow]Skipped:[/yellow] {old_file} (no SXXEYY pattern found)")
                    elif "Error" in status:
                        error_files.append((old_name, status))
                        progress.console.log(f"[red]Error renaming {old_file}:[/red] {status}")
                    elif status == "Skipped due to exit request":
                        skipped_files.append(old_name)
                        progress.console.log(f"[yellow]Skipped:[/yellow] {old_file} (exit requested)")
                    
                    if exit_requested:
                        # Cancel remaining tasks - we're shutting down
                        for f in futures:
                            f.cancel()
                        break
                        
                except Exception as e:
                    progress.console.log(f"[red]Error processing {old_file}:[/red] {e}")
                
                progress.update(task, advance=1)
                
                # Check for exit condition
                if exit_requested:
                    break

except KeyboardInterrupt:
    console.print("\n[bold yellow]Interrupt received. Exiting...[/bold yellow]")
except Exception as e:
    console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")
finally:
    # Final Output
    if renamed_files or skipped_files or error_files:
        console.print("\n[bold green]Renaming process completed![/bold green]")
        
        if renamed_files:
            console.print(f"[bold cyan]Successfully renamed {len(renamed_files)} files:[/bold cyan]")
            for old, new in renamed_files:
                console.print(f"[white]{old}[/white] -> [bold green]{new}[/bold green]")
        
        if skipped_files:
            console.print(f"[bold yellow]Skipped {len(skipped_files)} files (no pattern found or interrupted)[/bold yellow]")
        
        if error_files:
            console.print(f"[bold red]Encountered errors with {len(error_files)} files:[/bold red]")
            for file, error in error_files:
                console.print(f"[white]{file}[/white] -> [bold red]{error}[/bold red]")
    else:
        console.print("[yellow]No files were processed.[/yellow]")

    console.print(f"\n[bold green]Check the '{VIDEO_DIR}' directory for results.[/bold green]")

    if exit_requested:
        console.print("[yellow]Process was interrupted. Some files may not have been processed.[/yellow]")