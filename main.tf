provider "google" {
  credentials = file("~/.gcp/gcp-key.json")
  project     = "stellar-utility-170606"
  region      = "asia-east1"
}

resource "google_compute_instance" "default" {
  name         = "autobot-vm"
  machine_type = "e2-small"  # Cheapest machine type
  zone         = "asia-east1-a"  # Specify the zone within the region

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"  # Choose an image suitable for you
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Include this to give the instance a public IP address
    }
  }

  tags = ["http-server"]

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y git wget bzip2 supervisor

    # Clone the GitHub repository
    git clone https://github.com/enarjord/passivbot.git /opt/passivbot

    # Create directories for secrets and download the secrets files
    mkdir -p /opt/passivbot/secrets /opt/passivbot/test

    # Set proper permissions for the /opt/passivbot directory
    chown -R $(whoami) /opt/passivbot

    # Fetch the secrets files from metadata
    curl -o /opt/passivbot/api-keys.json http://metadata.google.internal/computeMetadata/v1/instance/attributes/secrets -H "Metadata-Flavor: Google"
    curl -o /opt/passivbot/test/live_config.json http://metadata.google.internal/computeMetadata/v1/instance/attributes/live_config -H "Metadata-Flavor: Google"

    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    export PATH="/opt/miniconda/bin:$PATH"

    # Initialize conda
    conda init bash
    source ~/.bashrc

    # Create a conda environment and activate it
    conda create -y -n passivbot-env python=3.9
    conda activate passivbot-env
[program:passivbot_KASUSDT]
command=/opt/passivbot/start_KASUSDT.sh
directory=/opt/passivbot
autostart=true
autorestart=true
stderr_logfile=/var/log/passivbot_KASUSDT.err.log
stdout_logfile=/var/log/passivbot_KASUSDT.out.log

    # Change directory to the app
    cd /opt/passivbot

    # Install the requirements using conda
    pip install -r requirements.txt

    # Ensure the script is executable
    chmod +x /opt/passivbot/start_passivbot.sh

    # Create Supervisor configuration file
    cat <<EOT > /etc/supervisor/conf.d/passivbot.conf
[program:passivbot]
command=/opt/passivbot/start_passivbot.sh
directory=/opt/passivbot
autostart=true
autorestart=true
stderr_logfile=/var/log/passivbot.err.log
stdout_logfile=/var/log/passivbot.out.log
EOT

    # Reload Supervisor to apply the new configuration
    sudo supervisorctl reread
    sudo supervisorctl update

    # Start the program via Supervisor
    # sudo supervisorctl start passivbot
  EOF

  metadata = {
    secrets     = file("api-keys.json")
    live_config = file("test/live_config.json")
  }
}

output "instance_ip" {
  value = google_compute_instance.default.network_interface.0.access_config.0.nat_ip
}
