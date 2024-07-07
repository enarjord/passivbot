#!/bin/bash

CONFIG_DIR="/opt/passivbot/configs/live"
SUPERVISOR_CONFIG_DIR="/etc/supervisor/conf.d"

# Clear existing Supervisor configurations for passivbot
sudo rm -f ${SUPERVISOR_CONFIG_DIR}/passivbot_*.conf

# Generate Supervisor configurations for each config file
for config_file in ${CONFIG_DIR}/*.json; do
    symbol=$(basename ${config_file} .json)
    script_path="/opt/passivbot/start_passivbot_${symbol}.sh"
    
    # Create a wrapper script for each symbol
    cat <<EOT > ${script_path}
#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 ${symbol} ${config_file} --leveraage 20
EOT

    # Ensure the script is executable
    chmod +x ${script_path}

    # Create Supervisor configuration file for each bot
    sudo bash -c "cat <<EOT > ${SUPERVISOR_CONFIG_DIR}/passivbot_${symbol}.conf
[program:passivbot_${symbol}]
command=${script_path}
directory=/opt/passivbot
autostart=true
autorestart=true
stderr_logfile=/var/log/passivbot_${symbol}.err.log
stdout_logfile=/var/log/passivbot_${symbol}.out.log
EOT"
done

# Reload Supervisor to apply the new configurations
sudo supervisorctl reread
sudo supervisorctl update
