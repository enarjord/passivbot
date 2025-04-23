import asyncio
import os
import time
import subprocess
import sys
import select
import platform

RUST_SOURCE_DIR = "passivbot-rust/"
COMPILED_EXTENSION_NAME = "libpassivbot_rust"
LOCK_FILE = os.path.join(RUST_SOURCE_DIR, ".compile.lock")
LOCK_TIMEOUT = 300  # 5 minutes in seconds
LOCK_CHECK_INTERVAL = 2  # Check every 2 seconds


def get_compiled_extension_paths():
    possible_extensions = ["so", "dylib", "dll", "pyd", "", "bundle", "sl"]
    return [
        os.path.join(RUST_SOURCE_DIR, "target", "release", f"{COMPILED_EXTENSION_NAME}.{ext}")
        for ext in possible_extensions
    ]


COMPILED_EXTENSION_PATHS = get_compiled_extension_paths()


def acquire_lock():
    start_time = time.time()
    while True:
        try:
            if os.path.exists(LOCK_FILE):
                # Check if lock is stale
                if time.time() - os.path.getmtime(LOCK_FILE) > LOCK_TIMEOUT:
                    print("Found stale lock file. Removing and proceeding with compilation.")
                    try:
                        os.remove(LOCK_FILE)
                    except OSError:
                        pass
                else:
                    if time.time() - start_time > LOCK_TIMEOUT:
                        print("Lock timeout reached. Assuming crashed compilation and proceeding.")
                        try:
                            os.remove(LOCK_FILE)
                        except OSError:
                            pass
                        return True
                    print(
                        f"Another compilation in progress. Waiting... ({int(time.time() - start_time)}s)"
                    )
                    time.sleep(LOCK_CHECK_INTERVAL)
                    continue

            # Create lock file
            with open(LOCK_FILE, "w") as f:
                f.write(str(os.getpid()))
            return True

        except OSError as e:
            print(f"Error managing lock file: {e}")
            return False


def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except OSError as e:
        print(f"Error removing lock file: {e}")


def check_compilation_needed():
    # Skip Rust compilation checks if requested via environment variable
    if os.environ.get("SKIP_RUST_COMPILE", "").lower() in ("1", "true", "yes"):
        print("SKIP_RUST_COMPILE set; skipping Rust extension compilation.")
        return False
    try:
        # Find the most recently modified compiled extension
        compiled_files = [path for path in COMPILED_EXTENSION_PATHS if os.path.exists(path)]
        if not compiled_files:
            print(f"No Rust extension found. Compiling...")
            return True  # No extension found, compilation needed

        compiled_time = max(os.path.getmtime(path) for path in compiled_files)

        # Check all .rs files in the Rust source directory
        for root, _, files in os.walk(RUST_SOURCE_DIR):
            for file in files:
                if file.endswith(".rs"):
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) > compiled_time:
                        print(f"Rust extension found, but out of date. Recompiling...")
                        return True  # A source file is newer, compilation needed
        return False  # No compilation needed
    except Exception as e:
        print(f"Error checking compilation status: {e}")
        return True  # If in doubt, suggest recompilation


def prompt_user_for_recompilation():
    print("Rust code needs recompilation. Recompile now? [Y/n]")
    start_time = time.time()
    while time.time() - start_time < 10:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            user_input = sys.stdin.readline().strip().lower()
            if user_input == "n":
                return False
            else:
                return True
    print("No input received within 10 seconds. Proceeding with recompilation.")
    return True


def recompile_rust():
    try:
        current_dir = os.getcwd()
        os.chdir(RUST_SOURCE_DIR)
        result = subprocess.run(
            ["maturin", "develop", "--release"], check=True, capture_output=True, text=True
        )
        os.chdir(current_dir)
        print("Compilation successful.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"An error occurred during compilation: {e}")
        return False


def manage_rust_compilation():
    if check_compilation_needed():
        if acquire_lock():
            try:
                if recompile_rust():
                    print("Rust extension successfully recompiled.")
                else:
                    print("Failed to recompile Rust extension. Please compile manually.")
                    sys.exit(1)
            finally:
                release_lock()
        else:
            print("Failed to acquire lock for compilation. Please try again later.")
            sys.exit(1)
    else:
        print("Rust extension is up to date.")


if __name__ == "__main__":
    manage_rust_compilation()
    from passivbot import main

    asyncio.run(main())
