import requests
import json


channel_list = {
                "pro" : 958078641483427880,
                "passivbot" : 926406999107846245, 
                "test" : 955193076668829696, 
                "onlyupx3" : 910612726081024010, 
                "tedybot" : 956956942633414817 
}

def get_channel_id(channel_code):
    return channel_list[channel_code]

def get_bot_commands_enabled_channels():
    return [
                get_channel_id('pro'),
                get_channel_id('test'),
                get_channel_id('tedybot')
    ]

def get_pro_channel_enabled():
    return [get_channel_id('pro'), get_channel_id('test')]

def send_slack_message(text, blocks = None):
    slack_token = open("config/token_slack.txt", 'r').read().strip()
    slack_channel = '#wallet'
    slack_icon_emoji = ':see_no_evil:'
    slack_user_name = 'WalletBot'
    print( requests.post('https://slack.com/api/chat.postMessage', {
        'token': slack_token,
        'channel': slack_channel,
        'text': text,
        'icon_emoji': slack_icon_emoji,
        'username': slack_user_name,
        'blocks': json.dumps(blocks) if blocks else None
    }).json())