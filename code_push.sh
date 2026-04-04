#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a commit message."
    echo 'Usage: ./code_push.sh "your commit message"'
    exit 1
fi

COMMIT_MSG=$1

echo "Adding changes..."
git add .

echo "Committing changes..."
git commit -m "$COMMIT_MSG"

echo "Pushing changes..."
git push

echo "Successfully pushed to repository!"
