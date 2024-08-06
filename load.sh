#!/bin/bash

# Set the chunk size
CHUNK_SIZE=100

# Get the list of files
FILES=$(find data/train_set1 -type f)

# Initialize counter
COUNTER=0

# Loop through the files and add them in chunks
for FILE in $FILES; do
  git add "$FILE"
  COUNTER=$((COUNTER + 1))

  # Commit after adding CHUNK_SIZE files
  if [ $COUNTER -ge $CHUNK_SIZE ]; then
    git commit -m "Add a chunk of $CHUNK_SIZE files"
    git push origin main
    COUNTER=0
  fi
done

# Commit and push any remaining files
if [ $COUNTER -gt 0 ]; then
  git commit -m "Add remaining files"
  git push origin main
fi