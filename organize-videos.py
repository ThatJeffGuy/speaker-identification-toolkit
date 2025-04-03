import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
import multiprocessing

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
    title = Text("VIDEO FILE RENAMING TOOL", style="bold cyan")
    console.print(Panel(title, border_style="cyan", expand=False, padding=(1, 2)))
    console.print("This tool renames video files by extracting their SxxEyy (Season/Episode) tags.")
    console.print("[italic yellow]Files will be renamed in place, keeping their original extensions.[/italic yellow]\n")

# Define Directory Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")

# Main Function
def main():
    # Display title
    print_title()
    
    # Ensure Directory Exists
    if not os.path.exists(VIDEO_DIR):
        console.print(f"[bold red]Error:[/bold red] Directory '{VIDEO_DIR}' not found.")
        console.print(f"[yellow]Creating directory: {VIDEO_DIR}[/yellow]")
        os.makedirs(VIDEO_DIR, exist_ok=True)
        console.print(f"[green]Please add video files to: {VIDEO_DIR}[/green]")
        return
    
    # Regex Pattern to Match SXXEYY
    pattern = re.compile(r"(S\d{2}E\d{2})", re.IGNORECASE)
    
    # Process Video Files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mkv", ".mp4", ".avi"))]
    if not video_files:
        console.print(f"[bold yellow]No video files found in '{VIDEO_DIR}'.[/bold yellow]")
        return
    
    # Ask for confirmation
    console.print(f"[bold green]Found {len(video_files)} video file(s).[/bold green]")
    proceed = console.input("[bold yellow]Do you want to proceed with renaming? (y/n): [/bold yellow]").strip().lower()
    if proceed != 'y':
        console.print("[yellow]Operation cancelled by user.[/yellow]")
        return
    
    console.print("[bold green]Starting renaming process...[/bold green]")
    
    # Threaded Function to Rename Files
    def rename_file(file):
        match = pattern.search(file)
        if match:
            new_name = f"{match.group(1)}{os.path.splitext(file)[1]}"  # Retain file extension
            old_path = os.path.join(VIDEO_DIR, file)
            new_path = os.path.join(VIDEO_DIR, new_name)
            
            # Check if target file already exists
            if os.path.exists(new_path) and old_path != new_path:
                return file, None, f"Target file {new_name} already exists"
            
            try:
                os.rename(old_path, new_path)
                return file, new_name, "Success"
            except Exception as e:
                return file, None, str(e)
        return file, None, "No SxxExx pattern found"
    
    # Multithreading with Rich Progress Bar
    MAX_THREADS = min(8, max(1, multiprocessing.cpu_count() - 1))
    console.print(f"[bold cyan]Using up to {MAX_THREADS} threads for renaming...[/bold cyan]")
    
    renamed_files = []
    skipped_files = []
    error_files = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
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
                    if new_name:
                        renamed_files.append((old_name, new_name))
                        progress.console.log(f"[green]Renamed:[/green] {old_name} -> [bold]{new_name}[/bold]")
                    elif "No SxxExx pattern" in status:
                        skipped_files.append((old_name, status))
                        progress.console.log(f"[yellow]Skipped:[/yellow] {old_file} ({status})")
                    else:
                        error_files.append((old_name, status))
                        progress.console.log(f"[red]Error renaming {old_file}:[/red] {status}")
                except Exception as e:
                    error_files.append((old_file, str(e)))
                    progress.console.log(f"[red]Error processing {old_file}:[/red] {e}")
                progress.update(task, advance=1)
    
    # Final Output
    console.print(f"\n{SEPARATOR}")
    console.print("[bold green]Renaming process complete![/bold green]")
    
    console.print(f"[bold cyan]Summary:[/bold cyan]")
    console.print(f"[green]Renamed: {len(renamed_files)} files[/green]")
    console.print(f"[yellow]Skipped: {len(skipped_files)} files[/yellow]")
    console.print(f"[red]Errors: {len(error_files)} files[/red]")
    
    if renamed_files:
        console.print("\n[bold cyan]Renamed Files:[/bold cyan]")
        for old, new in renamed_files:
            console.print(f"[white]{old}[/white] -> [bold green]{new}[/bold green]")
    
    console.print(f"\n[bold green]Check the '{VIDEO_DIR}' directory for results.[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")