# Resolving modify/delete merge conflicts

When you pull updates from `origin` you may see an error similar to:

```
CONFLICT (modify/delete): risk_management/configuration.py deleted in HEAD and modified in <commit>.
```

This happens when your local branch deleted a file that was updated on the branch you are pulling from. Git pauses the merge so you can decide which version to keep. Follow these steps to recover and finish the pull safely:

1. **Inspect Git status**

   ```bash
   git status
   ```

   The conflicted file will appear under the "unmerged paths" section. Git also leaves the other branch's version at `risk_management/configuration.py~<commit>` so you can review what changed.

2. **Decide which version should survive**

   * To keep the file exactly as it exists on the branch you pulled from, restore it with:

     ```bash
     git checkout --theirs risk_management/configuration.py
     ```

   * To keep your local deletion (removing the file completely), remove both copies:

     ```bash
     rm risk_management/configuration.py risk_management/configuration.py~*
     ```

   * To merge the two versions manually, open both files in an editor, copy the desired pieces into `risk_management/configuration.py`, and delete the backup file.

3. **Mark the conflict as resolved**

   After you have the correct contents in place:

   ```bash
   git add risk_management/configuration.py
   rm -f risk_management/configuration.py~*
   ```

4. **Commit the merge (if required)**

   Complete the pull operation:

   ```bash
   git commit
   ```

   Git will reuse the merge message from the pull.

5. **Verify your workspace**

   Run your test suite or start the application to confirm everything works before pushing the resolved merge.

Keeping a clean local working tree before running `git pull` (commit or stash your work) makes the resolution process smoother.
