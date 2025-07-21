#!/usr/bin/env python3
"""
Wrapper script to log all console outputs of another script to file.
Usage: python3 run_with_logging.py <command> [args...]
Example: python3 run_with_logging.py python3 src/main.py --config config.json
"""

import sys
import os
import subprocess
import signal
import re
from datetime import datetime, timezone
from pathlib import Path


def sanitize_filename(text):
    """Sanitize a string to be safe for use as a filename."""
    # Replace spaces and path separators with underscores
    text = re.sub(r"[\s/\\]", "_", text)
    # Remove or replace other problematic characters
    text = re.sub(r'[<>:"|?*]', "", text)
    # Remove leading/trailing dots and spaces
    text = text.strip(". ")
    # Limit length
    if len(text) > 100:
        text = text[:100]
    return text


def create_log_filename(command_args):
    """Create a log filename based on the command and timestamp."""
    # Join command arguments and sanitize
    command_str = " ".join(command_args)
    sanitized_command = sanitize_filename(command_str)

    # Add UTC timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Create filename with timestamp first
    log_filename = f"{timestamp}_{sanitized_command}.log"

    return log_filename


class LoggingWrapper:
    def __init__(self, command_args, log_file_path):
        self.command_args = command_args
        self.log_file_path = log_file_path
        self.process = None
        self.log_file = None

    def __enter__(self):
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        # Open log file
        self.log_file = open(self.log_file_path, "w", encoding="utf-8", buffering=1)

        # Write header to log file
        self.log_file.write(f"=== Log started at {datetime.now(timezone.utc).isoformat()} ===\n")
        self.log_file.write(f"Command: {' '.join(self.command_args)}\n")
        self.log_file.write("=" * 50 + "\n\n")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            self.log_file.write(f"\n=== Log ended at {datetime.now(timezone.utc).isoformat()} ===\n")
            self.log_file.close()

    def run(self):
        """Run the command and log all output."""
        try:
            # Start the process
            self.process = subprocess.Popen(
                self.command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Read and log output line by line
            for line in iter(self.process.stdout.readline, ""):
                # Write to console (preserve original behavior)
                sys.stdout.write(line)
                sys.stdout.flush()

                # Write to log file
                self.log_file.write(line)
                self.log_file.flush()

            # Wait for process to complete
            return_code = self.process.wait()
            return return_code

        except KeyboardInterrupt:
            # Forward the interrupt signal to the child process
            if self.process:
                print(f"\nForwarding keyboard interrupt to child process (PID: {self.process.pid})")
                self.log_file.write(
                    f"\n[WRAPPER] Keyboard interrupt received, forwarding to child process\n"
                )

                try:
                    # Send SIGINT to the child process
                    self.process.send_signal(signal.SIGINT)

                    # Wait a bit for graceful shutdown
                    try:
                        return_code = self.process.wait(timeout=10)
                        print(f"Child process exited gracefully with code: {return_code}")
                        self.log_file.write(
                            f"[WRAPPER] Child process exited gracefully with code: {return_code}\n"
                        )
                        return return_code
                    except subprocess.TimeoutExpired:
                        print("Child process didn't exit gracefully, sending SIGTERM")
                        self.log_file.write(
                            "[WRAPPER] Timeout waiting for graceful exit, sending SIGTERM\n"
                        )
                        self.process.terminate()

                        try:
                            return_code = self.process.wait(timeout=5)
                            return return_code
                        except subprocess.TimeoutExpired:
                            print("Child process still running, sending SIGKILL")
                            self.log_file.write(
                                "[WRAPPER] Timeout waiting for SIGTERM, sending SIGKILL\n"
                            )
                            self.process.kill()
                            return self.process.wait()

                except ProcessLookupError:
                    # Process already terminated
                    print("Child process already terminated")
                    self.log_file.write("[WRAPPER] Child process already terminated\n")
                    return self.process.returncode

            # Re-raise the KeyboardInterrupt
            raise


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_with_logging.py <command> [args...]", file=sys.stderr)
        print(
            "Example: python3 run_with_logging.py python3 src/main.py --config config.json",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get command arguments
    command_args = sys.argv[1:]

    # Create log filename
    log_filename = create_log_filename(command_args)
    log_file_path = Path("logs") / log_filename

    print(f"Logging output to: {log_file_path}")

    # Run with logging
    try:
        with LoggingWrapper(command_args, log_file_path) as wrapper:
            return_code = wrapper.run()
            print(f"\nProcess completed with return code: {return_code}")
            sys.exit(return_code)
    except KeyboardInterrupt:
        print("\nWrapper interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error running wrapper: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
