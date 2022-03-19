async def hello(message):
    discord_user_name = str(message.author)
    if "Sawyer911#8247" == discord_user_name :
        await message.reply('Hooo non pas lui :)', mention_author=True)
    elif "tedyptedto#9436" == discord_user_name:
        await message.reply('Oui maître ? Que puis-je pour vous ?', mention_author=True)
    else:
        await message.reply('Salut poto !', mention_author=True)