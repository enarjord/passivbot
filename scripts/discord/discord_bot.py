import discord
from actions.hello import hello
from actions.pumpdump import pumpdump
from actions.long_short import long_short
from actions.chart import chart
import os

# python3 -m pip install python-binance
# pip install discord
# pip install plotly
# pip install kaleido
# https://github.com/Rapptz/discord.py
#d doc du framework : https://discordpy.readthedocs.io/en/latest/api.html#discord.Member

class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        if message.content.startswith('!hello'):
            await hello(message)

        if message.content.startswith('!chart'):
            await chart(message)

        if message.content.startswith('!ls'):
            await long_short(message)

        if 'dump' in message.content:
            await pumpdump(message, 'dump')
        elif 'pump' in message.content:
            await pumpdump(message, 'pump')
            

        # Garé pour les infos
        # await message.channel.send('Hello {0.author.mention}'.format(message))
        # await message.channel.send('Hello {0.author.mention}'.format(message))
        # await message.channel.send('Hello {0.author}'.format(message))


client = MyClient()
base_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__))+'/')+'/'
client.run(open(base_dir+"/config/token.txt", 'r').read())