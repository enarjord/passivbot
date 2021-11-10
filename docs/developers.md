# Developers guide

If you're a developer and would like to contribute to Passivbot development, welcome!
There's a lot of active development going on, but there's always more to do and more great ideas to
implement. If you want to get in touch on ways you can contribute, or have a great idea and want to discuss
before starting to implement it, feel free to get in touch on [this Discord server](https://discord.gg/QAF2H2UmzZ).

## Environment Setup

### Installing Required Libraries
```shell
python -m pip install -e .[dev,tests,backtesting]
```

### Setting Up ``pre-commit``
```shell
pre-commit install --install-hooks
```

Now, you're ready to start contributing.

#### Typing And ``mypy``

Passivbot uses python type hints, although, in the past these were not always checked.
When you attempt to commit your code changes, as part of the ``pre-commit`` routines,
``mypy`` will run against the whole code base and you **will** see a lot of errors.
Cleaning up these errors is a huge undertaking, but please see if the code you're
contributing adds additional errors and **at least** fix those, please.

Once you've fixed any ``mypy`` errors you added, commit your code like:
```shell
env SKIP=mypy git commit <your regular git flags here>
```

## Pull requests

To add new functionality, you will need to create a fork on Github. After doing so, you can make the required changes
on a new feature branch. After pushing these to your fork, you can create a pull request to have the changes merged
into the original repository.

When you create a pull request, make sure you think about the following things:

* Are any changes needed to the documentation implemented?
* Is the functionality properly tested by yourself before offering it up for review?
* Did you set the right target branch on the pull request?
* If possible, write a short description of the change(s) offered, so it's easier for reviewers to understand

After creating your pull request, it will either be merged, or you will receive feedback on where to improve. Be
assured that any efforts are most appreciated, even if you receive feedback on things to improve!

## Pledges

If there is specific functionality that users would like to receive, they can pledge a bounty to whoever implements
this functionality properly in Passivbot. If you want to create a pledge, or get an overview of the current pledges
that have been made, join [this Discord server](https://discord.gg/QAF2H2UmzZ).

In order to create a pledge, just declare your pledge formally in the appropriate Discord channel, for example something like:

```
############

I hereby pledge a bounty of $X to be paid to whoever implements functionality Y. {optional description of the requested functionality}

May my reputation be tarnished in the eyes of the members of this here discord group should I not pay the promised $50 to the one who fulfils this bounty

############
```
