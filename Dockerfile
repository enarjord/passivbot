FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
# Skip runtime Rust compilation; extensions are prebuilt at image build time
ENV SKIP_RUST_COMPILE=true

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        pkg-config \
        libssl-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy only the package build inputs. Keep source entrypoints available for Compose.
COPY pyproject.toml setup.py MANIFEST.in rust-toolchain.toml ./
COPY requirements-dev.txt requirements-full.txt requirements-live.txt requirements-rust.txt ./
COPY schemas/config-v9.schema.json schemas/
COPY src src
COPY passivbot-rust/Cargo.toml passivbot-rust/Cargo.lock passivbot-rust/build.rs passivbot-rust/
COPY passivbot-rust/src passivbot-rust/src

# Install the full dependency set, including the existing developer tools.
RUN python -m pip install --no-cache-dir ".[dev]"

COPY broker_codes.hjson broker_codes.hjson

# Default command to run the bot; override with docker-compose command or docker run args
CMD ["python", "src/main.py"]
