# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
import time
import os

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
# import debugpy
# # Listen on a port (pick an unused one, e.g., 5678)
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # Execution stops here until VSCode attaches


base_path = "/mnt/scratch-artemis/margot/LoRe/PersonalLLM"

out_path_train = os.path.join(base_path, "embeddings", "train")
output_file_train = f"{out_path_train}.safetensors"

out_path_test = os.path.join(base_path, "embeddings", "test")
output_file_test = f"{out_path_test}.safetensors"

start_time = time.time()

print(f"[INFO] Working directory: {os.getcwd()}")
print(f"[INFO] Output files: {output_file_train}, {output_file_test}")


#train data embeddings generation
if not os.path.exists(output_file_train):
    # Load the dataset
    print("[INFO] Loading dataset...")
    dataset = load_dataset("namkoong-lab/PersonalLLM")
    data = pd.DataFrame(dataset["train"])
    print(f"[OK] Loaded {len(data)} training examples.")

    # Number of prompts
    num_prompts = len(data)
    dataset = []
    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        # Iterate over each response (1 to 8)
        prompt_responses = []
        for i in range(1,9):
            # Iterate over each model name
            column_name = f"response_{i}"
            conv = [{"role": "user", "content": row['prompt']}, {"role": "assistant", "content": row[column_name]}]
            prompt_responses.append(conv)
        dataset.append(prompt_responses)

    import torch

    from transformers import AutoModel, AutoTokenizer

    # Load model and tokenizer
    device = "cuda:0"
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    # attn_implementation="flash_attention_2",
        num_labels=1
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize lists to store embeddings and corresponding labels
    embeddings = []
    rewards = []
    print("[INFO] Generating embeddings...")


    # Iterate over each example in the dataset with a progress bar
    for idx, example in enumerate(dataset):
        if (idx + 1) % max(1, len(dataset) // 10) == 0:
            print(f"[INFO] Processed {idx + 1}/{len(dataset)} examples...")
        for i in range(len(example)):
            # print(example[i])
            inputs = rm_tokenizer.apply_chat_template(example[i], tokenize=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = rm(**inputs)  # Forward pass through the model
                # print(output)
                # Extract the last hidden state of the last token and move it to CPU
                # rewards.append(output.logits[0][0].item())
                embeddings.append(output.last_hidden_state[0][-1].cpu())
                # print(rewards[-1])
                # print(embeddings[-1])

    embeddings = torch.stack(embeddings, dim=0)  # Stack all embeddings into a single tensor
    print(f"[OK] Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    # rewards = torch.stack(rewards, dim=0)  # Stack all embeddings into a single tensor

    from safetensors.torch import save_file
    import os

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path_train), exist_ok=True)

    # Save the embeddings and labels in a safetensors file with shard indexing
    print(f"[INFO] Saving embeddings to {out_path_train}.safetensors...")
    save_file(
        {"embeddings": embeddings},
        f"{out_path_train}.safetensors",
    )


# Verify the file was created and has content
if os.path.exists(output_file_train):
    file_size_mb = os.path.getsize(output_file_train) / (1024 * 1024)
    print(f"[OK] Output file created: {output_file_train}")
    print(f"[OK] File size: {file_size_mb:.2f} MB")
else:
    raise RuntimeError(f"Output file was not created: {output_file_train}")



#test data embeddings generation
if not os.path.exists(output_file_test):

    # Load the dataset
    print("[INFO] Loading dataset...")
    dataset = load_dataset("namkoong-lab/PersonalLLM")
    data_test = pd.DataFrame(dataset["test"])
    print(f"[OK] Loaded {len(data_test)} test examples.")

    # Number of prompts
    num_prompts = len(data_test)
    dataset = []
    # Iterate over each row in the DataFrame
    for index, row in data_test.iterrows():
        # Iterate over each response (1 to 8)
        prompt_responses = []
        for i in range(1,9):
            # Iterate over each model name
            column_name = f"response_{i}"
            conv = [{"role": "user", "content": row['prompt']}, {"role": "assistant", "content": row[column_name]}]
            prompt_responses.append(conv)
        dataset.append(prompt_responses)

    import torch

    from transformers import AutoModel, AutoTokenizer

    # Load model and tokenizer
    device = "cuda:0"
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    # attn_implementation="flash_attention_2",
        num_labels=1
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize lists to store embeddings and corresponding labels
    embeddings = []
    rewards = []
    print("[INFO] Generating embeddings...")


    # Iterate over each example in the dataset with a progress bar
    for idx, example in enumerate(dataset):
        if (idx + 1) % max(1, len(dataset) // 10) == 0:
            print(f"[INFO] Processed {idx + 1}/{len(dataset)} examples...")
        for i in range(len(example)):
            # print(example[i])
            inputs = rm_tokenizer.apply_chat_template(example[i], tokenize=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = rm(**inputs)  # Forward pass through the model
                # print(output)
                # Extract the last hidden state of the last token and move it to CPU
                # rewards.append(output.logits[0][0].item())
                embeddings.append(output.last_hidden_state[0][-1].cpu())
                # print(rewards[-1])
                # print(embeddings[-1])

    embeddings = torch.stack(embeddings, dim=0)  # Stack all embeddings into a single tensor
    print(f"[OK] Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    # rewards = torch.stack(rewards, dim=0)  # Stack all embeddings into a single tensor

    from safetensors.torch import save_file
    import os

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path_test), exist_ok=True)

    # Save the embeddings and labels in a safetensors file with shard indexing
    print(f"[INFO] Saving embeddings to {out_path_test}.safetensors...")
    save_file(
        {"embeddings": embeddings},
        f"{out_path_test}.safetensors",
    )


# Verify the file was created and has content
if os.path.exists(output_file_test):
    file_size_mb = os.path.getsize(output_file_test) / (1024 * 1024)
    print(f"[OK] Output file created: {output_file_test}")
    print(f"[OK] File size: {file_size_mb:.2f} MB")
else:
    raise RuntimeError(f"Output file was not created: {output_file_test}")


elapsed_seconds = time.time() - start_time
print(f"[OK] Script completed successfully in {elapsed_seconds / 60:.2f} minutes.")