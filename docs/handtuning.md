# Tuning config by hand

The most advised way to create a profitable and resilient configuration is by using the automated
optimizer to find the best configuration. Besides this automated proces, there is also a hand-tuner
provided, which allows the user to manually tune & visualize a configuration.

## Running the hand-tuner

The hand-tuner is provided as a jupyter notebook, which you can easily run from a browser.
In order to run this, you first need to install jupyter labs using the following command:

```shell
pip3 install jupyterlab
```

After this is installed, you need to start jupyter from the terminal using the command `jupyter lab`.
This will open a browser window with jupyter. On the left hand side you will see a file tree.
By clicking on the file `hand_tuner.ipynb` you will open the hand tuner in the right side of the screen.

Before executing the hand-tuner, you need to fill in the appropriate account name in one of the cells.
Simply look for `self.user` the find the right place to put your account name.

After updating the account name, you can execute the hand-tuner by pressing the `play` icon at the
top of the screen. You can execute the individual steps manually one by one, or you can use `shift`
to run multiple steps at once.

## Tuning

Once you are able to run the entire hand-tuner script from the browser, you will see that the script
produces multiple plots and data along the way. In order to find the configuration you are after,
you can change the configuration parameters and rerun the script. The configuration parameters used
for plotting are exactly identical to a regular configuration file, which means can simply copy the
config file in the browser and run it. By making modifications to the parameters you will be able
to get a feel for the impact they have, and by inspecting the graphs you can change it to your needs