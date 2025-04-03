import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
import multiprocessing
import json

# Initialize Rich Console with a fixed width
CONSOLE_WIDTH = 50
console = Console(width=CONSOLE_WIDTH)

# Clear the terminal screen
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Constants
SEPARATOR = "=" * CONSOLE_WIDTH

# Title Screen
def print_title():
    title = Text("ENG AUDIO EXTRACTION TOOL", style="bold cyan")
    console.print(Panel(title, border_style="cyan", expand=False, padding=(1, 2)))
    console.print("This tool extracts only English audio tracks from video files.")
    console.print("[italic yellow]Audio is converted to mono 16-bit PCM at 44.1kHz for optimal processing.[/italic yellow]\n")

# Define Directory Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")

# Function to extract English audio with minimal metadata
def extract_eng_audio(file):
    input_path = os.path.join(VIDEO_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + ".wav")

    try:
        # Use ffprobe to parse JSON output for stream info
        ffprobe_command = [
            "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
            "stream=index:stream_tags=language", "-of", "json", input_path
        ]
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.stderr:
            return file, f"Error: {result.stderr}"
            
        stream_data = json.loads(result.stdout)

        # Find English audio stream index
        stream_index = None
        for stream in stream_data.get("streams", []):
            # Check for English language tag (eng, en, or english)
            lang = stream.get("tags", {}).get("language", "").lower()
            if lang in ["eng", "en", "english"]:
                stream_index = stream["index"]
                break

        if stream_index is None:
            return file, "No English audio track found"

        # Use FFmpeg to clean metadata and extract the English audio track
        command = [
            "ffmpeg", "-i", input_path,
            "-map", f"0:{stream_index}",  # Map the identified English audio stream
            "-acodec", "pcm_s16le",         # 16-bit PCM
            "-ar", "44100",                # 44.1kHz
            "-ac", "1",                    # Mono audio
            "-map_metadata", "-1",         # Strip all metadata
            "-y", output_path               # Overwrite existing file
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return file, "Success"
    except subprocess.CalledProcessError as e:
        return file, f"FFmpeg error: {e}"
    except json.JSONDecodeError as e:
        return file, f"JSON parsing error: {e}"
    except Exception as e:
        return file, f"Unexpected error: {e}"

def main():
    # Display title
    print_title()
    
    # Ensure Directories Exist
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # List Video Files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mkv", ".mp4", ".avi"))]
    if not video_files:
        console.print(f"[bold yellow]No video files found in '{VIDEO_DIR}'.[/bold yellow]")
        return
    
    # Ask for confirmation
    console.print(f"[bold green]Found {len(video_files)} video file(s).[/bold green]")
    proceed = console.input("[bold yellow]Do you want to proceed with audio extraction? (y/n): [/bold yellow]").strip().lower()
    if proceed != 'y':
        console.print("[yellow]Operation cancelled by user.[/yellow]")
        return
    
    console.print("[bold green]Starting extraction...[/bold green]")
    
    # Multithreaded Processing with Progress Bar
    MAX_THREADS = min(8, max(1, multiprocessing.cpu_count() - 1))
    console.print(f"[bold cyan]Using up to {MAX_THREADS} threads for processing...[/bold cyan]")
    
    results = []
    successful = []
    no_english = []
    errors = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Extracting English audio...", total=len(video_files))
    
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(extract_eng_audio, file): file for file in video_files}
            for future in as_completed(futures):
                file = futures[future]
                try:
                    original, status = future.result()
                    results.append((original, status))
                    
                    if status == "Success":
                        successful.append(original)
                        progress.console.log(f"[green]Extracted:[/green] {original}")
                    elif "No English audio track found" in status:
                        no_english.append(original)
                        progress.console.log(f"[yellow]{status}:[/yellow] {original}")
                    else:
                        errors.append((original, status))
                        progress.console.log(f"[red]{status}:[/red] {original}")
                except Exception as e:
                    errors.append((file, str(e)))
                    progress.console.log(f"[red]Error processing {file}:[/red] {e}")
                progress.update(task, advance=1)
    
    # Summary
    console.print(f"\n{SEPARATOR}")
    console.print("[bold green]Processing complete![/bold green]")
    
    console.print(f"[bold cyan]Summary:[/bold cyan]")
    console.print(f"[green]Successfully extracted: {len(successful)} files[/green]")
    console.print(f"[yellow]No English audio: {len(no_english)} files[/yellow]")
    console.print(f"[red]Errors: {len(errors)} files[/red]")
    
    if successful:
        console.print("\n[bold green]Successful extractions:[/bold green]")
        for file in successful:
            console.print(f"[white]{file}[/white] -> [cyan]{os.path.splitext(file)[0]}.wav[/cyan]")
    
    if errors:
        console.print("\n[bold red]Files with errors:[/bold red]")
        for file, error in errors:
            console.print(f"[white]{file}[/white] -> [red]{error}[/red]")
    
    console.print(f"\n[bold green]Check the '{OUTPUT_DIR}' directory for extracted audio files.[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")