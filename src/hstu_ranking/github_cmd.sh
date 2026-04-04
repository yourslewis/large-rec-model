# Switch to main branch
git checkout main
# Pull latest code
git pull origin main
# View all branches
git branch

# Create new branch based on latest code
git checkout -b hstu_ranking/single-task-binary
git checkout hstu_ranking/msan-toy-data
git checkout -b hstu_ranking/iterable_dataset

# Add changes (all files)
git add .
# Commit changes
git commit -m "add single task binary classification for hstu ranking"
git commit -m "Upate ranker for msan toy data. Task: binary. Features: seq_id, event_id, action_weights(pseudo_0 + target)"
git commit -m "Modify the dataset for iterably reading streaming data (with buffered shuffle) from cosmos db."
# Push to remote repository
git push origin hstu_ranking/single-task-binary
git push origin hstu_ranking/msan-toy-data
git push origin hstu_ranking/iterable_dataset



# Ignore all data files
**/*.csv
**/*.tsv
**/*.json
**/*.parquet
**/*.dat

# Ignore all model files
**/*.pth
**/*.pt
**/*.ckpt
**/*.h5
**/*.distcp

# Ignore all log files
**/*.log
**/*.out


# Undo last commit but keep file modifications (back to state after git add):
git reset --soft HEAD~1
# Undo commit and also don't keep git add state (back to state before git add):
git reset --mixed HEAD~1


# Check if ssh agent is running
ssh-add -l
# If not running, start ssh agent
eval "$(ssh-agent -s)"
# Add SSH security key to agent
ssh-add ~/.ssh/id_ed25519
