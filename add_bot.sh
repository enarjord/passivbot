#!/bin/bash

# Ensure the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <symbol> <config-file>"
    exit 1
fi

SYMBOL=$1
CONFIG_FILE=$2
CONFIG_DIR="/opt/passivbot/configs/live"

# Copy the new configuration file to the config directory
cp ${CONFIG_FILE} ${CONFIG_DIR}/${SYMBOL}.json

# Run the dynamic Supervisor configuration script
/opt/passivbot/update_supervisor_configs.sh
