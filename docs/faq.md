# Frequently asked questions

**Q: Do I have to know how to code to use this?**  
A: No. While knowing how to code may be useful in understanding how the bot works, you do not need to "code" anything yourself. Effectively using the bot does require some forward thinking tech ability.

**Q: Can I edit the code?**  
A: Yes! PassivBot is open source for anybody to use, redistribute, or modify without exception or restriction. If you accomplish something, let us know! Or don't! It's up to you.

**Q: My API keys aren't working!**  
A: Make sure you've setup your account correctly with the exchange, and opened your futures account prior to generating your API Keys. Some users have reported that API keys generated prior to successfully opening the "futures" portion of their account are rendered useless due to account protection protocols enforced by your exchange.

**Q: Why does the bot cancel and create orders so much?**  
A: Bot calculates ideal orders and compares them to actual orders.  Matching orders will be kept, deviating orders will be deleted and missing orders will be created.  Bot checks if orders are correct about once every 5 sec or less.  It's not a problem if bot cancels and creates often, as long as exchange's rate limit isn't exceeded.  Since initial entries are based on moving averages, entry prices may change often.

**Q: How much should I invest with the bot?**  
A: Never more than you are willing to lose. While backtests may show good results, and many have achieved high ROI using PassivBot, liquidations or soft stops do happen and are unpredictable. You alone are responsible for how you utilize the bot.

**Q: Can I trade another currency myself while the bot is running on my account?**  
A: We don't suggest it. PassivBot treats the balance of your futures wallet as if it inherits the sum. Even if you set your 'balance' lower in the configuration, the bot could be thrown out of order. Creating a new account generally seems like the best option, especially for tracking P/L.

**Q: So is my backtest going to be accurate?**  
A: Backtesting makes the assumption that forthcoming price action will be largely similar to the price history that was tested. Additionally, the backtest is unable to account for some factors such as errant market makers, partially filled orders, or increasing volume's effect on market prices. Backtesting is meant to provide insight into how a particular configuration behaves under the tested time period. Often, wildly profitable strategies may appear in a backtest, but fail to yield the same results in live testing due to "over-fitting". For this reason, we generally consider conservative settings to be the most reliable.

**Q: Should I intervene to help my bot sometimes?**  
A: This is largely a matter of personal choice.  Adding extra capital, opening/closing orders with market orders are all fine, bot will adjust itself.  Depending on your settings, the bot may be able to get out of a tricky position by itself, but if it is unable it could self-liquidate. If you find yourself having to manually intervene too often; check your config.

**Q: How do I make "Conservative" settings if I get my configurations from the backtester?**  
A: Firstly, the backtester provides some options to discard overly risky configurations. This is the best method of searching for conservative, sustainably profitable configurations. Additionally, if you did not make your own configuration, you can simply lower the balance parameters to specify how much of your margin balance is being used in trades. You can specify a percentage of your balance (this is the total amount the bot can use as margin), where the unused portion will serve to pad your liquidation price. The caveat to this method is that it will lower the profit efficiency of your settings as only a portion of every profit taken is used to compound into the next trade.

**Q: What if I can't leave the bot running on my computer all the time?**  
A: Many users choose to offload the program to a VPS such as Amazon EC2, Digital Ocean, or Vultr for 24/7 access. Generally, this beats out running the bot from your local machine as there is a lower possibility of downtime or dropped connections. Additionally, this allows specification of server region providing for a lower latency connection to your exchange. Theoretically, this results in a better preforming bot.

**Q: Can I "set it and forget it"?**  
A: Yes, but it's not advisable. Checking the bot's progress regularly and taking profits from the account has proven the best strategy to maximize long term returns. Occasionally a bot might experience errors or a dropped connection. While it is setup to re-start automatically, extenuating circumstances can sometimes prevent this from happening.

**Q: I can't (or don't want to) backtest; Where can I find good pre-made configurations?**  
A: A small repository of community tested configurations are available with their backtest results here, sorted by version number. Do note these configurations may or may not have been tested live, so use them with discretion.

**Q: I want to help with development, but don't know where to start.**  
A: Join our discord to get involved with the community!

**Q: My bot liquidated me, what now?**  
A: As PassivBot is still under development, it can do some strange things on occasion. As such, we suggest not using any money you cannot afford to lose. If you have trade data, notice an anomaly, or need some help figuring out what went wrong, post your data to the Discord and/or Telegram. The community is extremely helpful and may be able to provide some guidance.

**Q: I'm getting an error message stating 'Signature is invalid for this request', along with 'maxNotionalValue'?**  
A: The Binance API key that is used was created before Futures was enabled for the account. In order to fix this, you'll need to create a new API key.

**Q: I'm getting an error while installing requirements.txt stating 'No matching distribution found for ray\[tune\]==1.2.0'**  
A: Make sure you are using python 3.8.x
