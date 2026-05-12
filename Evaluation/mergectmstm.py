#!/usr/bin/env python
import sys

# -------------------------------------------------------------------------
# ROBUST MERGE SCRIPT for "ID Speaker Gloss" Format
# -------------------------------------------------------------------------
# This script ensures that every sentence ID in your STM (Ground Truth)
# exists in your CTM (Predictions). If a prediction is missing, it
# adds an empty placeholder so the evaluation tool doesn't crash.
# -------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python mergectmstm.py <ctm_file> <stm_file>")
        sys.exit(1)

    ctm_file = sys.argv[1]
    stm_file = sys.argv[2]

    # 1. Read the STM file (Ground Truth)
    # Format: "ID Speaker Gloss..."
    # We only care about the first column (ID) to ensure it exists in predictions.
    stm_ids = set()
    try:
        with open(stm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    # parts[0] is your "11August_2010_Wednesday_tagesschau-2"
                    stm_ids.add(parts[0])
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find STM file: {stm_file}")
        sys.exit(1)

    # 2. Read the CTM file (Your Model's Predictions)
    ctm_lines = []
    seen_ids = set()
    try:
        with open(ctm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    ctm_lines.append(parts)
                    seen_ids.add(parts[0])
    except FileNotFoundError:
        print(f"Warning: CTM file {ctm_file} not found. Creating a new one with empty entries.")

    # 3. Add Missing Entries (The Fix)
    # If the model missed a sentence completely, we add an [EMPTY] line
    # so SCLITE counts it as a "Deletion" error instead of crashing.
    missing_ids = stm_ids - seen_ids
    
    for missing_id in missing_ids:
        # Format: ID 1 0.000 0.030 [EMPTY]
        # This tells the eval tool: "There was a sentence here, but we predicted nothing."
        new_entry = [missing_id, "1", "0.000", "0.030", "[EMPTY]"]
        ctm_lines.append(new_entry)

    # 4. Sort and Save
    # Evaluation tools strictly require files to be sorted by ID.
    ctm_lines.sort(key=lambda x: x[0])

    with open(ctm_file, 'w') as f:
        for line in ctm_lines:
            f.write(" ".join(line) + "\n")

if __name__ == "__main__":
    main()
