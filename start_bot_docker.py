import os
import sys

exchange = sys.argv[1]


print(f"\nStarting Docker {exchange}")
with open('./api_key_secrets/{exchange}/passivbot.json'.format(exchange=exchange), 'w+') as f:
    f.truncate(0)
    f.write(
        '["{API_KEY}", "{API_SECRET}"]'.format(
            API_KEY=os.getenv("API_KEY"),
            API_SECRET=os.getenv("API_SECRET"),
        )
    )

with open('{0}'.format(os.getenv('CONFIG_FILE')), 'r') as from_file, \
        open('./live_settings/{exchange}/passivbot.json'.format(exchange=exchange), 'w+') as to_file:
    to_file.truncate(0)
    for line in from_file:
        to_file.write(line)

os.system('python3 start_bot.py {exchange} passivbot'.format(exchange=exchange))
