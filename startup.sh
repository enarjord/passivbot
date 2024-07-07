#!/bin/bash
apt-get update
apt-get install -y git wget bzip2 supervisor

# Clone the GitHub repository
git clone https://github.com/enarjord/passivbot.git /opt/passivbot

# Create directories for secrets and download the secrets files
mkdir -p /opt/passivbot/secrets /opt/passivbot/test /opt/passivbot/configs/live

# Set proper permissions for the /opt/passivbot directory
chown -R $(whoami) /opt/passivbot

# Fetch the secrets files from metadata
curl -o /opt/passivbot/secrets/secrets.json http://metadata.google.internal/computeMetadata/v1/instance/attributes/secrets -H "Metadata-Flavor: Google"
curl -o /opt/passivbot/test/live_config.json http://metadata.google.internal/computeMetadata/v1/instance/attributes/live_config -H "Metadata-Flavor: Google"

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda
export PATH="/opt/miniconda/bin:$PATH"

# Initialize conda
conda init bash
source ~/.bashrc

# Create a conda environment and activate it
conda create -y -n passivbot-env python=3.10
conda activate passivbot-env

# Change directory to the app
cd /opt/passivbot

# Install the requirements using conda
pip install -r requirements.txt

# Create the dynamic Supervisor configuration script
cat <<EOT > /opt/passivbot/update_supervisor_configs.sh
$(cat update_supervisor_configs.sh)
EOT

chmod +x /opt/passivbot/update_supervisor_configs.sh
