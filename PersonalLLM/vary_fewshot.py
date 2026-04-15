
import os
from safetensors.torch import load_file
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch as torch
import sys
import os
import gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = "/mnt/scratch-artemis/margot/LoRe/PersonalLLM"
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import *
import matplotlib.pyplot as plt
from datetime import datetime

from transformers import AutoModel, AutoTokenizer

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# Load model and tokenizer
log("Starting train_basis.py")
log("Loading model and tokenizer...")
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
   # attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize a variable to store the last linear layer
last_linear_layer = None
# Iterate over the model's modules
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
# Print the weights and bias of the last linear layer
V_final = last_linear_layer.weight[:,0].to(device).to(torch.float32).reshape(-1, 1)
log("Model loaded successfully")

# Load the dataset
log("Loading dataset...")
dataset = load_dataset("namkoong-lab/PersonalLLM")
data = pd.DataFrame(dataset["train"])
data_test = pd.DataFrame(dataset["test"])
log(f"Dataset loaded: {len(data)} train samples, {len(data_test)} test samples")

# Load the embeddings
log("Loading train embeddings...")
embeddings = load_file(f"{data_dir}/embeddings/train.safetensors")
embeddings = embeddings['embeddings']

features = []
# Number of prompts
num_prompts = len(data)
log(f"Processing {num_prompts} train prompts...")
for i in range(num_prompts):
    temp = []
    for j in range(8):
        temp.append(embeddings[i * 8 + j])
    features.append(temp)
    if (i + 1) % 50 == 0:
        log(f"  Processed {i+1}/{num_prompts} prompts")
log("Train features ready")

# Free memory
del embeddings
gc.collect()

log("Loading test embeddings...")
embeddings = load_file(f"{data_dir}/embeddings/test.safetensors")
embeddings = embeddings['embeddings']

test_features = []
# Number of prompts
num_prompts = len(data_test)
log(f"Processing {num_prompts} test prompts...")
for i in range(num_prompts):
    temp = []
    for j in range(8):
        temp.append(embeddings[i * 8 + j])
    test_features.append(temp)
    if (i + 1) % 50 == 0:
        log(f"  Processed {i+1}/{num_prompts} prompts")
log("Test features ready")

# Free memory
del embeddings
import gc
gc.collect()

# Generate features with the off the shelf reward models
# Define the list of model names corresponding to the columns
model_names = [
    "gemma_2b",
    "gemma_7b",
    "mistral_raft",
    "llama3_sfairx",
    "oasst_deberta_v3",
    "beaver_7b",
    "oasst_pythia_7b",
    "oasst_pythia_1b",
    "mistral_ray",
    "mistral_weqweasdas",
]

# Initialize a 3D NumPy array to store the reward tensors for each prompt
log(f"Building reward tensor for {len(data)} train prompts...")
num_prompts = len(data)
reward_tensor = np.empty((num_prompts, 8, len(model_names)), dtype=object)
# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    if (index + 1) % 50 == 0:
        log(f"  Processed {index+1}/{num_prompts} train prompts")
    # Create an 8x10 array for the current prompt
    prompt_array = np.empty((8, len(model_names)), dtype=object)

    # Iterate over each response (1 to 8)
    for i in range(1, 9):
        # Iterate over each model name
        for j, model_name in enumerate(model_names):
            # Construct the column name for the current response and model
            column_name = f"response_{i}_{model_name}"

            # Assign the value to the appropriate position in the prompt array
            if column_name in row:
                prompt_array[i - 1, j] = row[column_name]
            else:
                prompt_array[i - 1, j] = None  # or handle missing columns as needed

    # Assign the prompt array to the reward tensor
    reward_tensor[index] = prompt_array
# Now, reward_tensor is a 3D NumPy array with shape (num_prompts, 8, 10)
log("Train reward tensor complete")

# Initialize a 3D NumPy array to store the reward tensors for each prompt
log(f"Building reward tensor for {len(data_test)} test prompts...")
num_prompts = len(data_test)
reward_tensor_test = np.empty((num_prompts, 8, len(model_names)), dtype=object)
# Iterate over each row in the DataFrame
for index, row in data_test.iterrows():
    if (index + 1) % 50 == 0:
        log(f"  Processed {index+1}/{num_prompts} test prompts")
    # Create an 8x10 array for the current prompt
    prompt_array = np.empty((8, len(model_names)), dtype=object)

    # Iterate over each response (1 to 8)
    for i in range(1, 9):
        # Iterate over each model name
        for j, model_name in enumerate(model_names):
            # Construct the column name for the current response and model
            column_name = f"response_{i}_{model_name}"

            # Assign the value to the appropriate position in the prompt array
            if column_name in row:
                prompt_array[i - 1, j] = row[column_name]
            else:
                prompt_array[i - 1, j] = None  # or handle missing columns as needed

    # Assign the prompt array to the reward tensor
    reward_tensor_test[index] = prompt_array
# Now, reward_tensor is a 3D NumPy array with shape (num_prompts, 8, 10)
log("Test reward tensor complete")

# Free memory early
log("Cleaning up large arrays from memory...")
del data_test
del data
import gc
gc.collect()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}")


# Prepare Train Dataset
log("Preparing training data...")
N = 100
alpha_val = 0.001
alpha = alpha_val * np.ones(len(model_names))
log(f"Generating population with N={N}...")
W = generate_popupulation(alpha, N)
# Users at train time
log("Simulating seen users on train prompts...")
all_feature_diff = simulate_population(reward_tensor, features, W)
log("Creating sparse tensor for train features...")
train_features = create_sparse_tensor(all_feature_diff, 0.005)
del all_feature_diff
gc.collect()

# Seen users, unseen prompts
log("Simulating seen users on test prompts...")
all_feature_diff_test = simulate_population(reward_tensor_test, test_features, W)
log("Creating sparse tensor for test features...")
test_features_sparse = create_sparse_tensor(all_feature_diff_test, 1.0)
del all_feature_diff_test
gc.collect()

# Unseen Users
N_unseen = 50
log(f"Generating unseen population with N={N_unseen}...")
W_unseen = generate_popupulation(alpha, N_unseen)


# Unseen Users, Few Shot prompts
log("Simulating unseen users on train prompts...")
all_feature_diff_unseen = simulate_population(reward_tensor, features, W_unseen)
log("Creating sparse tensor for unseen train features...")
train_features_unseen = create_sparse_tensor(all_feature_diff_unseen, 0.05)# correspond à 500 ex par users
del all_feature_diff_unseen
gc.collect()

# Unseen Users, unseen prompts
log("Simulating unseen users on test prompts...")
all_feature_diff_test_unseen = simulate_population(reward_tensor_test, test_features, W_unseen)
log("Creating sparse tensor for unseen test features...")
test_features_sparse_unseen = create_sparse_tensor(all_feature_diff_test_unseen, 1.0)
del all_feature_diff_test_unseen
gc.collect()


# K_list = [5]
# alpha_list = [0]
# trials = 10
# num_shots = [1,3,5,10,15,20,25,50]#[5 * (i + 1) for i in range(3)]
# few_shot_train_accuracies_few_shot, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot_std = run_few_shot_vary_shots(trials, alpha_list, K_list, num_shots, train_features, train_features_unseen, test_features_sparse_unseen, V_final, N, N_unseen, device)

# # Plotting
# plt.figure(figsize=(8, 5))
# # Convert lists to numpy arrays for arithmetic operations
# accuracies = np.array(unseen_user_unseen_prompts_accuracies_few_shot)
# stds = np.array(unseen_user_unseen_prompts_accuracies_few_shot_std)
# plt.fill_between(num_shots, 
#                  accuracies - stds,
#                  accuracies + stds,
#                  alpha=0.2, label="±1 std")
# plt.plot(num_shots, accuracies, marker='o', linestyle='-', label="Seen Users")
# # plt.plot(K_list, train_accuracies_joint, marker='o', linestyle='-', label="Train Seen Users")
# # plt.plot(K_list, few_shot_train_accuracies_few_shot, marker='o', linestyle='-', label="Train Unseen Users Fewshot")
# plt.xlabel('nb example')
# plt.ylabel('Accuracies')
# run_signature = f"N={N}, N_unseen={N_unseen}, α={alpha_val}, ts={datetime.now().strftime('%Y%m%d_%H%M%S')}"
# plt.title(f'Generalization Accuracy vs. number examples \n({run_signature})')
# plt.xticks(num_shots, labels=["ref" if k==0 else str(k) for k in num_shots])
# plt.legend()

# alpha = alpha_list[0]
# # Save the plot
# plt.savefig(f'{SCRIPT_DIR}/generalization_accuracy_vs_nb_examples_lore_alpha_{alpha}.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()


K_list = [5]#[0, 1, 2, 3, 4, 5,6,7,8,9,10,15,20]
alpha_list = [0]
alpha_val = alpha_list[0]
trials = 10
num_shots = [5 * (i + 1) for i in range(35)]
few_shot_train_accuracies_few_shot_means_list, few_shot_train_accuracies_few_shot_stds_list, unseen_user_unseen_prompts_accuracies_few_shot_means_list, unseen_user_unseen_prompts_accuracies_few_shot_stds_list,accuracies_train_list, accuracies_test_list = run_few_shot_vary_shots(trials, alpha_list, K_list, num_shots, train_seen_data,test_seen_data, train_unseen_data, test_unseen_data, V_final, N, N_unseen, device)
print(few_shot_train_accuracies_few_shot_means_list, few_shot_train_accuracies_few_shot_stds_list, unseen_user_unseen_prompts_accuracies_few_shot_means_list, unseen_user_unseen_prompts_accuracies_few_shot_stds_list,accuracies_train_list, accuracies_test_list)

# Plotting
plt.figure(figsize=(8, 5))

# Convert lists to numpy arrays for arithmetic operations

if len(num_shots) > 1 and len(K_list)==1:
    accuracies_unseen = np.array(unseen_user_unseen_prompts_accuracies_few_shot_means_list[0])
    stds_unseen = np.array(unseen_user_unseen_prompts_accuracies_few_shot_stds_list[0])

    accuracies_unseen_train = np.array(few_shot_train_accuracies_few_shot_means_list[0])
    stds_unseen_train = np.array(few_shot_train_accuracies_few_shot_stds_list[0])
    
    accuracies_seen_train = np.array(accuracies_train_list*len(num_shots)) 
    accuracies_seen_test = np.array(accuracies_test_list*len(num_shots))
    plt.fill_between(num_shots, 
                    accuracies_unseen - stds_unseen,
                    accuracies_unseen + stds_unseen,
                    alpha=0.2, label="±1 std")
    plt.plot(num_shots, accuracies_unseen, marker='o', linestyle='-', label="Seen Users")
    plt.plot(num_shots, accuracies_unseen_train, marker='o', linestyle='-', label="Train Unseen Users")
    plt.plot(num_shots, accuracies_seen_train, marker='o', linestyle='-', label="Train Seen Users")
    
    # plt.plot(num_shots, accuracies_seen_test, marker='o', linestyle='-', label="Test Seen Users")
    # plt.plot(K_list, train_accuracies_joint, marker='o', linestyle='-', label="Train Seen Users")
    # plt.plot(K_list, few_shot_train_accuracies_few_shot, marker='o', linestyle='-', label="Train Unseen Users Fewshot")
    plt.xlabel('nb example')
    plt.ylabel('Accuracies')
    run_signature = f"N={N}, N_unseen={N_unseen}, α={alpha_val}, ts={datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plt.title(f'Generalization Accuracy vs. number examples \n({run_signature})')
    plt.xticks(num_shots, labels=["ref" if k==0 else str(k) for k in num_shots], rotation=45)
    plt.legend()
    plt.tight_layout()

    alpha = alpha_list[0]
    # Save the plot
    plt.savefig(f'{SCRIPT_DIR}/generalization_accuracy_vs_nb_examples_lore_alpha_{alpha}_ts={datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if len(K_list) > 1:
    accuracies_seen_train = np.array(accuracies_train_list)
    accuracies_seen_test = np.array(accuracies_test_list)

    accuracies_unseen = np.array(unseen_user_unseen_prompts_accuracies_few_shot_means_list).squeeze()
    stds_unseen = np.array(unseen_user_unseen_prompts_accuracies_few_shot_stds_list).squeeze()

    accuracies_unseen_train = np.array(few_shot_train_accuracies_few_shot_means_list).squeeze()
    stds_unseen_train = np.array(few_shot_train_accuracies_few_shot_stds_list).squeeze()
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_list, accuracies_seen_train, marker='o', linestyle='-', label="Train Seen Users")
    plt.plot(K_list, accuracies_seen_test, marker='o', linestyle='-', label="Test Seen Users")
    # plt.fill_between(K_list, 
    #                 accuracies_unseen - stds_unseen,
    #                 accuracies_unseen + stds_unseen,
    #                 alpha=0.2, label="±1 std")
    # plt.plot(K_list, accuracies_unseen, marker='o', linestyle='-', label="Seen Users")    
    
    plt.xlabel('rank')
    plt.ylabel('Accuracies')
    run_signature = f"N={N}, N_unseen={N_unseen}, α={alpha_val}, ts={datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plt.title(f'Generalization Accuracy vs. Rank\n({run_signature})')
    plt.xticks(K_list, labels=["ref" if k==0 else str(k) for k in K_list])
    plt.legend()
    plt.tight_layout()

    alpha = alpha_list[0]
    # Save the plot
    plt.savefig(f'{SCRIPT_DIR}/generalization_accuracy_vs_rank_lore_alpha_{alpha}_ts={datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()