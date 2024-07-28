#!/bin/bash

CONFIG_DIR="/opt/passivbot/configs/live"
SUPERVISOR_CONFIG_DIR="/etc/supervisor/conf.d"

# Clear existing Supervisor configurations for passivbot
sudo rm -f ${SUPERVISOR_CONFIG_DIR}/passivbot_*.conf

# Generate Supervisor configurations for each config file
for config_file in ${CONFIG_DIR}/*.json; do
    # Extract symbol and exchange from the filename
    filename=$(basename ${config_file} .json)
    symbol=$(echo ${filename} | cut -d'_' -f1)
    exchange=$(echo ${filename} | cut -d'_' -f2)

    # Default to binance_01 if exchange is not provided
    if [ -z "${exchange}" ]; then
        exchange="binance_01"
    else
        exchange="${exchange}_01"
    fi

    # Construct the script path
    script_path="/opt/passivbot/start_passivbot_${symbol}_${exchange}.sh"
    
    # Create a wrapper script for each symbol and exchange
    cat <<EOT > ${script_path}
#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py ${exchange} ${symbol} ${config_file} --leverage 30
EOT

    # Ensure the script is executable
    chmod +x ${script_path}

    # Create Supervisor configuration file for each bot
    sudo bash -c "cat <<EOT > ${SUPERVISOR_CONFIG_DIR}/passivbot_${symbol}_${exchange}.conf
[program:passivbot_${symbol}_${exchange}]
command=${script_path}
directory=/opt/passivbot
autostart=true
autorestart=true
stderr_logfile=/var/log/passivbot_${symbol}_${exchange}.err.log
stdout_logfile=/var/log/passivbot_${symbol}_${exchange}.out.log
EOT"
done

# Reload Supervisor to apply the new configurations
sudo supervisorctl reread
sudo supervisorctl update
