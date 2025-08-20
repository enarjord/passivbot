#!/bin/zsh -e
# Stages and commits each unstaged tracked file separately.
# - Does NOT include untracked files.
# - Commit message format: "Update <file_path>"
#
# Suggested usage:
# chmod +x .assistant/commit_unstaged.sh
# ./.assistant/commit_unstaged.sh

# Collect unstaged file paths (handles paths with spaces)
any=false
while IFS= read -r -d '' f; do
  any=true
  # Ensure the file is a tracked path (skip untracked)
  if git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
    echo "Staging and committing: $f"
    git add -- "$f"
    git commit -m "Update $f"
  else
    echo "Skipping untracked or missing file: $f"
  fi
done < <(git diff --name-only -z)

if [ "$any" = false ]; then
  echo "No unstaged changes found."
fi
