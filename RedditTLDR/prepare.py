# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
import pandas as pd
import numpy as np
import gc
import os
import pickle
import time
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "tldr_embeddings_train.pkl")
start_time = time.time()

print(f"[INFO] Working directory: {os.getcwd()}")
print(f"[INFO] Output file: {out_path}")

if os.path.exists(out_path):
    print(f"[INFO] Output already exists, skipping embedding generation: {out_path}")
    with open(out_path, "rb") as f:
        results = pickle.load(f)
    worker_count = len(results)
    pair_count = sum(len(items) for items in results.values())
    print(f"[OK] Loaded cached embeddings for {worker_count} workers and {pair_count} pairs.")
    print("[OK] Script completed successfully.")
    exit(0)

# # Load the dataset using HuggingFace Hub API directly (bypass load_dataset script issue)
# print("[INFO] Downloading dataset from HuggingFace Hub...")
# try:
#     # Try to download parquet file directly
#     parquet_file = hf_hub_download(
#         repo_id="openai/summarize_from_feedback",
#         filename="axis/validation/0.parquet",
#         repo_type="dataset"
#     )
#     print(f"[INFO] Downloaded parquet: {parquet_file}")
#     df = pd.read_parquet(parquet_file)
# except Exception as e:
#     print(f"[WARN] Parquet download failed ({e}), trying alternative format...")
#     # Fallback: try JSON files
#     try:
#         json_file = hf_hub_download(
#             repo_id="openai/summarize_from_feedback",
#             filename="data/train-00000-of-00001.jsonl",
#             repo_type="dataset"
#         )
#         print(f"[INFO] Downloaded JSON: {json_file}")
#         df = pd.read_json(json_file, lines=True)
#     except Exception as e2:
#         print(f"[ERROR] Both parquet and JSON downloads failed: {e2}")
#         print("[ERROR] Falling back to load_dataset (may fail)...")
dataset = load_dataset("openai/summarize_from_feedback", 'comparisons')
df = pd.DataFrame(dataset['train'])

print(f"[INFO] Loaded train split with {len(df)} rows.")

# Create an empty dictionary to store the results
worker_results = {}
# Iterate over each row in the dataset
for index, row in df.iterrows():
    # Get the worker ID
    worker_id = row['worker']
    
    # Get the text, winning summary, and losing summary
    text = row['info']['post']
    summaries = row['summaries']
    winning_summary = summaries[row['choice']]['text']
    losing_summary = summaries[1 - row['choice']]['text']
    
    # Add the result to the worker's list
    if worker_id not in worker_results:
        worker_results[worker_id] = []
    worker_results[worker_id].append({
        'text': text,
        'winning_summary': winning_summary,
        'losing_summary': losing_summary
    })

# Sort the worker_results dictionary by the number of entries
worker_results = dict(sorted(worker_results.items(), key=lambda item: len(item[1]), reverse=False))
total_pairs_expected = sum(len(items) for items in worker_results.values())
print(f"[INFO] Grouped data into {len(worker_results)} workers and {total_pairs_expected} pairs.")
# for i in range(20, len(all_worker_results), 1):
# worker_results = dict(list(all_worker_results.items())[1:2])

from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"[INFO] Loaded reward model on device {device}: {model_name}")

results = {}
for worker_id, data in worker_results.items():
    # print(worker_id)
    results[worker_id] = []
    # Print the results
    correct_predictions = 0
    total_predictions = 0
    print(f"Worker ID: {worker_id}")
    for entry in data:
        conv_winning = [{"role": "user", "content": entry['text']}, {"role": "assistant", "content": entry['winning_summary']}]
        conv_losing = [{"role": "user", "content": entry['text']}, {"role": "assistant", "content": entry['losing_summary']}]
        inputs_winning = rm_tokenizer.apply_chat_template(conv_winning, return_tensors="pt").to(device)
        inputs_losing = rm_tokenizer.apply_chat_template(conv_losing, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding_winning = rm(inputs_winning).last_hidden_state[0][-1].cpu()
            embedding_losing = rm(inputs_losing).last_hidden_state[0][-1].cpu()

        results[worker_id].append({
            'text': entry['text'],
            'winning_summary': entry['winning_summary'],
            'losing_summary': entry['losing_summary'],
            'embeddings': {
                'winning': [embedding_winning],
                'losing': [embedding_losing]
            }
        })

# with open('tldr_embeddings_val.pkl', 'wb') as f:
with open(out_path, 'wb') as f:
    pickle.dump(results, f)

worker_count = len(results)
pair_count = sum(len(items) for items in results.values())
if pair_count != total_pairs_expected:
    raise RuntimeError(
        f"Mismatch in generated pairs: expected {total_pairs_expected}, got {pair_count}."
    )
if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
    raise RuntimeError(f"Output file missing or empty: {out_path}")

elapsed_seconds = time.time() - start_time
print(f"[OK] Saved embeddings for {worker_count} workers and {pair_count} pairs.")
print(f"[OK] File size: {os.path.getsize(out_path) / (1024 * 1024):.2f} MB")
print(f"[OK] Finished in {elapsed_seconds / 60:.2f} minutes.")
print("[OK] Script completed successfully.")
