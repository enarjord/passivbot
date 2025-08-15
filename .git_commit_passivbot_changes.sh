#!/bin/zsh
# Commits staged changes for the specified files with descriptive messages.
# Run this script from the repository root to create the commits.

# Commit changes to backtest.rs
git add passivbot-rust/src/backtest.rs
git commit -m "chore(backtest): tidy backtest.rs — remove unused imports and silence warnings"

# Commit changes to closes.rs
git add passivbot-rust/src/closes.rs
git commit -m "chore(closes): tidy closes.rs — minor cleanup and formatting"
