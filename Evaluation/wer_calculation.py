import os
import re

def clean_hypothesis(text):
    """
    Safely processes CTC model outputs for the PHOENIX-2014T raw dataset.
    Performs critical sequence deduplication without altering the native vocabulary.
    """
    if not text:
        return ""
    
    # 1. Standardize spacing
    text = ' '.join(text.split())
    
    # 2. CTC Sequence Deduplication
    tokens = text.split()
    if not tokens:
        return ""
        
    cleaned_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        if tokens[i] != tokens[i-1]:
            cleaned_tokens.append(tokens[i])
            
    return ' '.join(cleaned_tokens)

def calculate_edit_distance(ref, hyp):
    """Calculates Levenshtein distance between two lists of words."""
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref)][len(hyp)]

def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None, triplet=False):

    print("\n--- Running Pure Python WER Evaluation ---")

    pred_file = prefix + output_file
    gt_file = os.path.join(evaluate_dir, evaluate_prefix + "-" + mode + ".stm")

    preds = {}
    gts = {}

    # 1. Read Ground Truth (Format: ID Speaker Gloss1 Gloss2...)
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Ensure we have at least ID, Speaker, and one Gloss
                if len(parts) > 2:
                    vid_id = parts[0]
                    # parts[1] is the Speaker (e.g., Signer08), so we read from parts[2] onwards
                    raw_text = " ".join(parts[2:]) 
                    # We do NOT clean the ground truth vocabulary!
                    gts[vid_id] = raw_text.split()
    except FileNotFoundError:
        print(f"Error: Could not find Ground Truth file at {gt_file}")
        return 100.0

    # 2. Read Predictions (Auto-Detect Format)
    try:
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                vid_id = parts[0]

                # Format A: Strict NIST CTM
                if len(parts) >= 5 and parts[1] in ["1", "A"] and parts[2].replace('.', '', 1).isdigit():
                    word = parts[4]
                    if word != "[EMPTY]":
                        word = ' '.join(word.split()) 
                        if vid_id not in preds:
                            preds[vid_id] = []
                        if word: 
                            preds[vid_id].append(word)

                # Format B: Standard Sentence Format (ID Word1 Word2...)
                else:
                    if len(parts) > 1:
                        raw_text = " ".join(parts[1:])
                        # Clean the hypothesis to remove CTC duplicates
                        cleaned_text = clean_hypothesis(raw_text)
                        preds[vid_id] = cleaned_text.split()
                    else:
                        preds[vid_id] = []

    except FileNotFoundError:
        print(f"Error: Could not find Prediction file at {pred_file}")
        return 100.0

    # 3. Calculate Word Error Rate
    total_errors = 0
    total_words = 0

    for vid_id, gt_words in gts.items():
        pred_words = preds.get(vid_id, [])
        errors = calculate_edit_distance(gt_words, pred_words)
        total_errors += errors
        total_words += len(gt_words)

    if total_words == 0:
        return 100.0

    final_wer = (total_errors / total_words) * 100.0
    print(f"[{mode.upper()}] Python Calculated WER: {final_wer:.2f}%")
    print("------------------------------------------\n")

    return final_wer
