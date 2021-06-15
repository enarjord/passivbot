#Analyzing results

PassivBot includes some code to assist with analyzing test results & bot performance. This allows users to visualize trade activity, sizes and duration along with some additional metrics to determine stability and intended usage for a given configuration. To use this part of the code, you’ll need to have Jupyter-lab installed:

`pip3 install jupyterlab`

Next, open a command line / terminal and navigate to the PassivBot root directory.. You can then open the Jupyter Interface by running:

`jupyter lab`

The Jupyter Interface should open, and display the PassivBot folder in the toolbar on the left. Navigate to the file ‘backtesting_notes.ipynb’. The ‘backtesting_notes.ipynb’ file only needs a single change before you can get the results of your backtest. Under cell 5 of the file, find and change the symbol to the currency pair your backtesting results are for. For example, if we are backtesting Ethereum:

`symbol = 'ETHUSDT'`

Save the file, and return focus to the first cell. You may now use ‘shift + enter’ to step through the code sequentially, or hit the ‘play’ icon. Ensure each step finishes before proceeding to run the next step. The cell ID will change from an asterisk [*] to the ID [1] after finishing.

This Information can then be amputated into custom scripts, manipulated to display new metrics, or saved as files.
