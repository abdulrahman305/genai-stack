#!/bin/sh

# Fetch the latest changes from the upstream repository
git fetch upstream

# Merge the changes from the upstream repository into the current branch
git merge upstream/main

# Push the merged changes to the origin repository
git push origin main
