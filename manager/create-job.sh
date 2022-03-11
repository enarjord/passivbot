#!/bin/bash

if [ "$EUID" -ne 0 ]
then echo "Please run as root"
    exit
fi


SERVICES=/etc/systemd/system
SERVICE_NAME=passivbot-manager.service
MANAGER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PASSIVBOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

# delete if arg -d is passed
if [ "$1" == "-d" ]
then
    echo "Deleting job"
    sudo systemctl disable $SERVICE_NAME
    sudo rm -f $SERVICES/$SERVICE_NAME
    exit
fi

echo "This script will create a systemctl service"
echo "to run configured instances on system start"
echo
echo "Do you want to continue? (y/n)"
read -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted"
    exit
fi

echo "Creating service file..."
cat <<EOF > $SERVICES/$SERVICE_NAME
[Unit]
Description=Start passivbot instances on boot
After=network.target

[Service]
Type=simple
User=root
ExecStart=su $SUDO_USER -c "python3 manager start -ay"
WorkingDirectory=$PASSIVBOT_DIR
Restart=no

[Install]
WantedBy=multi-user.target
EOF

echo "Enabling service..."
systemctl enable passivbot-manager.service
echo "Done"
echo
echo "Run this script with -d to delete the job"