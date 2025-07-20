#!/usr/bin/env python3
"""
Logging wrapper for main.py that captures all output to a timestamped log file.
Usage: python3 run_with_logging.py {original_args}
"""

import os
import sys
import subprocess
import datetime
import re
from pathlib import Path


def sanitize_filename(text):
    """Convert text to filename-safe string."""
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r"[^\w\-_.]", "_", text)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Limit length to avoid filesystem issues
    return sanitized[:100] if len(sanitized) > 100 else sanitized


def create_log_filename(script_name, args):
    """Create a descriptive, timestamped log filename."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get base script name without path and extension
    base_name = Path(script_name).stem

    # Create args string (limit length and sanitize)
    if args:
        args_str = "_".join(args)
        args_str = sanitize_filename(args_str)
        if len(args_str) > 50:  # Keep args portion reasonable
            args_str = args_str[:47] + "..."
        filename = f"{base_name}_{args_str}_{timestamp}.log"
    else:
        filename = f"{base_name}_{timestamp}.log"

    return filename


def setup_log_directory():
    """Create logs directory if it doesn't exist."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def run_with_logging():
    """Run the main script with comprehensive logging."""
    if len(sys.argv) < 1:
        print("Error: No arguments provided")
        sys.exit(1)

    # Set up logging
    log_dir = setup_log_directory()
    script_path = "src/main.py"
    args = sys.argv[1:]  # Skip the wrapper script name

    log_filename = create_log_filename(script_path, args)
    log_path = log_dir / log_filename

    # Build the command
    cmd = [sys.executable, script_path] + args
    cmd_str = " ".join(cmd)

    print(f"Running: {cmd_str}")
    print(f"Logging to: {log_path}")
    print("-" * 60)

    # Write initial log entry
    with open(log_path, "w") as log_file:
        log_file.write(f"=== Passivbot Execution Log ===\n")
        log_file.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        log_file.write(f"Command: {cmd_str}\n")
        log_file.write(f"Working Directory: {os.getcwd()}\n")
        log_file.write(f"Python Version: {sys.version}\n")
        log_file.write("=" * 50 + "\n\n")

    try:
        # Run the process with real-time output capture
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1
        )

        # Stream output to both console and file
        with open(log_path, "a") as log_file:
            for line in iter(process.stdout.readline, ""):
                # Print to console
                print(line, end="")
                sys.stdout.flush()

                # Write to log file with timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"[{timestamp}] {line}")
                log_file.flush()

        # Wait for process to complete
        return_code = process.wait()

        # Log completion
        with open(log_path, "a") as log_file:
            log_file.write(f"\n{'='*50}\n")
            log_file.write(f"Process completed at: {datetime.datetime.now().isoformat()}\n")
            log_file.write(f"Exit code: {return_code}\n")

        print(f"\nExecution completed. Full log saved to: {log_path}")
        return return_code

    except KeyboardInterrupt:
        print(f"\nExecution interrupted. Partial log saved to: {log_path}")
        if "process" in locals():
            process.terminate()
        return 130
    except Exception as e:
        error_msg = f"Error running script: {e}"
        print(error_msg)

        # Log the error
        with open(log_path, "a") as log_file:
            log_file.write(f"\n{'='*50}\n")
            log_file.write(f"ERROR at {datetime.datetime.now().isoformat()}: {error_msg}\n")

        return 1


if __name__ == "__main__":
    exit_code = run_with_logging()
    sys.exit(exit_code)
