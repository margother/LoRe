# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from urllib import response

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_user(reward_tensor, features, w):
    num_prompts = len(reward_tensor)
    feature_diff = []
    for i in range(num_prompts):
        scores = np.dot(reward_tensor[i], w)
        # Find the index of the response with the largest and smallest score
        largest_score_index = np.argmax(scores)
        smallest_score_index = np.argmin(scores)
        feature_diff.append(features[i][largest_score_index] - features[i][smallest_score_index])
    feature_diff = torch.stack(feature_diff, dim=0)
    return feature_diff

def evaluate_model(X, V, w):
    # Compute the expression X @ V @ w
    X = torch.tensor(X, dtype=torch.float32)
    # result = X @ V @ w
    result = X @ V @ w
    # Count the number of positive elements
    num_positive = (result > 0).sum().item()
    # Compute the fraction of positive elements
    fraction_positive = num_positive / result.numel()
    return fraction_positive

def simulate_population(reward_tensor, features, W):
    all_feature_diff = [simulate_user(reward_tensor, features, w) for w in W]
    return torch.stack(all_feature_diff, dim=0)

def generate_popupulation(alpha, N):
    return np.random.dirichlet(alpha, N)

def create_sparse_tensor(dense_tensor, sample_percentage):
    """
    Creates a sparse tensor by randomly sampling entries from a dense tensor.
    Args:
        dense_tensor (torch.Tensor): The input dense tensor.
        sample_percentage (float): The percentage of entries to sample per row.
    Returns:
        torch.Tensor: The resulting sparse tensor.
    """
    # Get the shape of the dense tensor
    N, M, d = dense_tensor.shape
    # Calculate the number of samples per row
    num_samples_per_row = int(sample_percentage * M)
    # Create a list to store the sparse rows
    sparse_rows = []
    # Iterate over each row of the dense tensor
    for i in range(N):
        # Randomly select indices for sampling
        indices = np.random.choice(M, num_samples_per_row, replace=False)
        
        # Sample values from the dense tensor
        values = dense_tensor[i, indices]
        
        # Append the sampled values to the list of sparse rows
        # sparse_rows.append(values.to(device))
        sparse_rows.append(torch.tensor(values, dtype=torch.float32).to(device))
    return sparse_rows

def create_dataset_prism(embeddings):
    sparse_rows = []
    for user_id, dialogs in embeddings.items():
        values = None
        for dialog_id, examples in dialogs.items():
            for i in range(len(examples["chosen"])):
                chosen = torch.tensor(embeddings[user_id][dialog_id]["chosen"][i], dtype=torch.float32, device=device)
                rejected = torch.tensor(embeddings[user_id][dialog_id]["rejected"][i], dtype=torch.float32, device=device)
                diff = (chosen - rejected).reshape(1, -1)
                if values is None:
                    values = diff
                else:
                    values = torch.cat((values, diff), dim=0)
        sparse_rows.append(values)
    return sparse_rows

def create_dataset_prism_shots(embeddings, shots):
    sparse_rows = []
    for user_id, dialogs in embeddings.items():
        values = None
        idx = random.sample([i for i in range(len(dialogs))], shots)
        j = 0
        for dialog_id, examples in dialogs.items():
            if j in idx:
                for i in range(len(examples["chosen"])):
                    chosen = torch.tensor(embeddings[user_id][dialog_id]["chosen"][i], dtype=torch.float32, device=device)
                    rejected = torch.tensor(embeddings[user_id][dialog_id]["rejected"][i], dtype=torch.float32, device=device)
                    diff = (chosen - rejected).reshape(1, -1)
                    if values is None:
                        values = diff
                    else:
                        values = torch.cat((values, diff), dim=0)
            j += 1       
        sparse_rows.append(values)
    return sparse_rows

def learn_multiple(train_features, num_iterations=1000, learning_rate=0.01):
    W_list = []
    V_list = []
    num_features = train_features[0][0].shape[0]
    N = len(train_features)
    for i in range(N):
        am = AlternatingMinimization(1, num_features, 1, num_iterations, learning_rate).to(device)
        w, V = am.train([train_features[i]])
        W_list.append(w[0])
        V_list.append(V.detach())
    return W_list, V_list

def learn_multiple_few_shot(train_features, V, num_iterations=1000, learning_rate=0.01):
    N = len(train_features)
    num_features = train_features[0][0].shape[0]
    fitw = PersonalizeBatch(N, num_features, V.shape[1], num_iterations, learning_rate).to(device)
    W = fitw.train(train_features, V)
    return W

def learn_multiple_few_shot_weighted(alpha, train_features, current_dialog_features, V, num_iterations=1000, learning_rate=0.01):
    N = len(train_features)
    num_features = train_features[0][0].shape[0]
    fitw = PersonalizeBatch_weighted(alpha, N, num_features, V.shape[1], num_iterations, learning_rate).to(device)
    # W = [fitw.train([train_features[i]], V)[0] for i in range(N)] 
    W = fitw.train(train_features, current_dialog_features, V)
    return W

def eval_multiple(W_list, V_list, test_features):
    accuracies = []
    N = len(test_features)
    accuracies = [evaluate_model(test_features[i], V_list[i], W_list[i]) for i in range(N)]
    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Average accuracy: {average_accuracy:.4f}")
    print(f"Standard deviation of accuracy: {std_accuracy:.4f}")
    return accuracies

def solve(train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = train_features[0][0].shape[0]
    am = AlternatingMinimization(num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()

def solve_regularized(V_sft, alpha, train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = train_features[0][0].shape[0]
    am = LoRe(V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()

def solve_regularized_simplex(V_sft, alpha, train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = 4096
    am = LoRe_regularized(V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()

def solve_multi_reward(train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = train_features[0][0].shape[0]
    rm = MultiRewardModel(num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    rm.train(train_features)
    return rm

def learn_single_reward(train_features, num_iterations=1000, learning_rate=0.01):
    num_features = train_features[0][0].shape[0]
    N = len(train_features)
    sm = SingleRewardModel(N, num_features, 1, num_iterations, learning_rate).to(device)
    V = sm.train(train_features)
    return V.detach()

def learn_single_reward_regularized(V_ref, alpha, train_features, num_iterations=1000, learning_rate=0.01):
    num_features = train_features[0][0].shape[0]
    N = len(train_features)
    sm = SingleRewardModel_regularized(V_ref, alpha, N, num_features, 1, num_iterations, learning_rate).to(device)
    V = sm.train(train_features)
    return V.detach()

class LoRe_regularized(nn.Module):
    def __init__(
        self, V_sft, alpha, num_classes, num_features, num_basis_vectors,
        num_iterations, learning_rate
    ):
        super().__init__()
        self.V_sft = V_sft.to(device)
        self.V_sft_norm = F.normalize(self.V_sft, dim=0)   # normalize once
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.W = nn.Parameter(torch.rand(num_classes, num_basis_vectors, device=device))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors, device=device))

    # --- NEW: pack once ---
    @staticmethod
    def _prepare_batch(X):
        """
        X: list of length C; X[i] is [m_i, F]
        Returns:
          X_cat: [N, F], y: [N] (values in 0..C-1)
        """
        x_list, y_list = [], []
        for i, x in enumerate(X):
            x_list.append(x)
            y_list.append(torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
        X_cat = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return X_cat, y

    def _forward_from_packed(self, X_cat, y, alpha_curr):
        # choose which parameter set to freeze for this pass
        V_used = self.V
        # V_used = F.normalize(self.V, dim=0)
        W_logits = self.W

        W_row = F.softmax(W_logits, dim=1)    # [C, B]
        Vw    = V_used @ W_row.T              # [F, C]

        logits_all = (X_cat @ Vw) / 100.0     # [N, C]
        logits = logits_all.gather(1, y.unsqueeze(1)).squeeze(1)
        nll = -F.logsigmoid(logits).mean()

        # V-alignment reg should only act when we're updating V
        reg = 0.0
        if alpha_curr > 0:   
            V_norm = F.normalize(self.V, dim=0)
            V_sft_norm = F.normalize(self.V_sft, dim=0)
            cos_sim = (V_norm * V_sft_norm).sum(dim=0)
            reg = torch.mean(1 - cos_sim)

        # # Diversity reg should only act when we're updating W
        entropy_loss = 0.0
        # entropy_loss = self._diversity_loss_rows()

        return nll, reg, entropy_loss


    # keep a compatibility wrapper (packs every call)
    def forward(self, X, alpha_curr):
        X_cat, y = self._prepare_batch(X)
        return self._forward_from_packed(X_cat, y, alpha_curr)

    def _alpha_at_step(self, step: int) -> float:
        warmup_start = int(0.2 * self.num_iterations)
        warmup_end   = int(0.8 * self.num_iterations)
        if step < warmup_start: return 0.0
        if step >= warmup_end:  return float(self.alpha)
        return float(self.alpha) * (step - warmup_start) / (warmup_end - warmup_start)

    def train(self, X):
        self.to(device)
        X_cat, y = self._prepare_batch(X)
        X_cat = X_cat.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)

        optimizer_W = optim.Adam([self.W], lr=self.learning_rate)
        optimizer_V = optim.Adam([self.V], lr=self.learning_rate)

        for step in range(self.num_iterations):
            alpha_curr = self._alpha_at_step(step)

            # ---- Update W: freeze V ----
            optimizer_W.zero_grad()
            nll_W, _, _ = self._forward_from_packed(
                X_cat, y, alpha_curr=0.0)
            
            # loss_W = nll_W + self.entropy_weight * entropy_loss
            nll_W.backward()
            optimizer_W.step()

            # ---- Update V: freeze W ----
            optimizer_V.zero_grad()
            nll_V, reg, _ = self._forward_from_packed(
                X_cat, y, alpha_curr=alpha_curr
            )
            total_loss_V = nll_V + alpha_curr * reg
            total_loss_V.backward()
            optimizer_V.step()

            if (step + 1) == self.num_iterations:
                W_sm = F.softmax(self.W, dim=1)
                print(f"W mean per dim: {W_sm.mean(dim=0).detach().cpu().numpy()}")
                print(f"W std  per dim: {W_sm.std(dim=0).detach().cpu().numpy()}")
                # L2 norms of V columns (parameter) and of the normalized V used in forward
                with torch.no_grad():
                    V_param_norms = torch.linalg.vector_norm(self.V, ord=2, dim=0)
                print(f"||V[:, i]|| (param): {V_param_norms.detach().cpu().numpy()}")

                print(
                    f"Step {step}: "
                    f"NLL(W)={nll_W.item():.4f}, "
                    f"NLL(V)={nll_V.item():.4f}, "
                    f"Reg={float(reg):.4f}, "
                    f"Alpha={alpha_curr:.4f}, "
                )
        
        # ---- Return only directions with min_c softmax(W)[c, i] >= 1e-2 ----
        W_probs = F.softmax(self.W, dim=1)                   # [C, B]
        max_per_basis = W_probs.max(dim=0).values            # [B]
        print(max_per_basis)
        mask = (max_per_basis >= 1e-2)                       # bool[B]

        W_kept = W_probs[:, mask]                            # [C, B_kept]
        V_kept = self.V[:, mask]                             # [F, B_kept]
        num_kept = int(mask.sum().item())
        print(f"Num dimensions kept: {num_kept}/{self.num_basis_vectors} (threshold=1e-2)")

        print(f"W mean per dim: {W_kept.mean(dim=0).detach().cpu().numpy()}")
        print(f"W std  per dim: {W_kept.std(dim=0).detach().cpu().numpy()}")

        return W_kept, V_kept
                
        # return F.softmax(self.W, dim=1), self.V

class LoRe(nn.Module):
    def __init__(self, V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate):
        super(LoRe, self).__init__()
        self.V_sft = V_sft
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        # Initialize weight vectors and matrix V
        # self.w = [nn.Parameter(torch.randn(num_basis_vectors)) for _ in range(num_classes)]
        self.W = nn.Parameter(torch.randn(num_classes, num_basis_vectors))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors))
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, X):
        nll = 0 
        # V_w = self.V @ (torch.stack(self.w).T).to(device)
        # V_w = self.V @ (self.W).T 
        V_w = self.V @ (F.softmax(self.W, dim=1)).T 
        # Compute the log-likelihood function
        i = 0
        for x in X:
            # x = torch.tensor(x, dtype=torch.float32)
            # logits =  x @ V_w[:,i] / 100.0
            logits =  x @ V_w[:,i] / 100.0
            # print(logits)
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll +=  ((-log_likelihood.sum()) / len(x))
            # if self.alpha > 0:
            #     probs = F.softmax(self.W[i,:])
            #     entropy = -torch.sum(probs * torch.log(probs))
            #     nll += self.alpha * entropy
            i += 1
        
        reg = 0
        if self.alpha > 0:
            for j in range(self.num_basis_vectors):
                # print(self.V[:,j].shape)
                # print(self.V_sft.shape)
                reg += self.alpha * torch.sum((self.V[:,j] - self.V_sft)**2)
        return nll, reg
    
    
    def train(self, x):
        # Move the model and data to the GPU
        self.to(device)
        # x = [torch.tensor(i, dtype=torch.float32).to(device) for i in x]
        # Train the model using alternating minimization
        for j in range(self.num_iterations):
            # print("Iter : ", j)
            # Update weight vectors
            # for i in range(len(x)):
            self.optimizer.zero_grad()
            loss, reg = self.forward(x)
            regularized_loss = loss + reg
            regularized_loss.backward()
            self.optimizer.step()
            # print(loss.item())
            # if j % 100 == 0:
            #     print("Iter : ", j)
            #     print(loss.item())
        
        # return self.w, self.V
        # return self.W, self.V
        return (F.softmax(self.W, dim=1)), self.V

class PersonalizeBatch(nn.Module):
    def __init__(self, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate):
        super(PersonalizeBatch, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        # Initialize weight vectors and matrix V
        self.w = nn.ParameterList([nn.Parameter(torch.randn(num_basis_vectors)) for _ in range(num_classes)])
        
        # print(self.parameters())
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, X, V):
        nll = 0 
        # Compute the log-likelihood function
        i = 0
        for x in X:
            # V_w = V @ self.w[i]
            V_w = V @ F.softmax(self.w[i]) 
            # x = torch.tensor(x, dtype=torch.float32)
            logits =  x @ V_w / 100.0
            # print(logits)
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll +=  ((-log_likelihood.sum()) / len(x))
            i += 1
        return nll
    
    def train(self, X, V):
        # Train the model using alternating minimization
        for j in range(self.num_iterations):
            
            # Update weight vectors
            # for i in range(len(x)):
            self.optimizer.zero_grad()
            loss = self.forward(X, V)
            loss.backward()
            self.optimizer.step()
            # if j % 100 == 0:
            #     print("Iter : ", j)
            #     print(loss.item())
        
        return [F.softmax(self.w[i]).detach() for i in range(len(X))]

def run(K_list, alpha_list, V_final, train_features, test_features_sparse, 
                       train_features_unseen, test_features_sparse_unseen, N, N_unseen, device):
    """
    Compute accuracies for joint and few-shot learning.

    Parameters:
    K_list (list): List of values for K.
    alpha_list (list): List of values for alpha.
    V_final (tensor): Final value of V.
    train_features (tensor): Training features.
    test_features_sparse (tensor): Test features for seen users.
    train_features_unseen (tensor): Training features for unseen users.
    test_features_sparse_unseen (tensor): Test features for unseen users.
    N (int): Number of seen users.
    N_unseen (int): Number of unseen users.
    device (device): Device to use for computations.

    Returns:
    tuple: Tuple containing 9 numpy arrays with computed accuracies and standard deviations.
    """

    # Initialize lists to store results
    train_accuracies_joint = []
    seen_user_unseen_prompts_accuracies_joint = []
    few_shot_train_accuracies_few_shot = []
    unseen_user_unseen_prompts_accuracies_few_shot = []
    train_accuracies_joint_std = []
    seen_user_unseen_prompts_accuracies_joint_std = []
    few_shot_train_accuracies_few_shot_std = []
    unseen_user_unseen_prompts_accuracies_few_shot_std = []

    for alpha in alpha_list:
        # print("alpha : ", alpha)

        # Joint Reward and Weights Learning
        for K in K_list:
            print("K : ", K)
            if K == 0:
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
            else: 
                W_joint, V_joint = solve_regularized(V_final, alpha, train_features, K, num_iterations=1000, learning_rate=0.5)

            print("Train Performance")
            accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)
            train_accuracies_joint.append(np.mean(accuracies_train))
            train_accuracies_joint_std.append(np.std(accuracies_train))

            print("Seen User Unseen Prompts")
            accuracies_seen_user_unseen_prompts = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], test_features_sparse)
            seen_user_unseen_prompts_accuracies_joint.append(np.mean(accuracies_seen_user_unseen_prompts))
            seen_user_unseen_prompts_accuracies_joint_std.append(np.std(accuracies_seen_user_unseen_prompts))

            # Learn the w on unseen users with few shot interactions
            if K <= 1:
                W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
            else:
                W_few_shot = learn_multiple_few_shot(train_features_unseen, V_joint.detach(), num_iterations=500, learning_rate=0.1)

            print("Few Shot Train Performance")
            accuracies_few_shot_train = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen)
            few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
            few_shot_train_accuracies_few_shot_std.append(np.std(accuracies_few_shot_train))

            print("Unseen User Unseen Prompts")
            accuracies_unseen_user_unseen_prompts = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen)
            unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_user_unseen_prompts))
            unseen_user_unseen_prompts_accuracies_few_shot_std.append(np.std(accuracies_unseen_user_unseen_prompts))

    fac = 0.25
    train_accuracies_joint = np.array(train_accuracies_joint)
    seen_user_unseen_prompts_accuracies_joint = np.array(seen_user_unseen_prompts_accuracies_joint)
    few_shot_train_accuracies_few_shot = np.array(few_shot_train_accuracies_few_shot)
    unseen_user_unseen_prompts_accuracies_few_shot = np.array(unseen_user_unseen_prompts_accuracies_few_shot)
    train_accuracies_joint_std = fac * np.array(train_accuracies_joint_std)
    seen_user_unseen_prompts_accuracies_joint_std = fac * np.array(seen_user_unseen_prompts_accuracies_joint_std)
    few_shot_train_accuracies_few_shot_std = fac * np.array(few_shot_train_accuracies_few_shot_std)
    unseen_user_unseen_prompts_accuracies_few_shot_std = fac * np.array(unseen_user_unseen_prompts_accuracies_few_shot_std)

    return train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std


def sample_shots(train_features_unseen, shots):
    """
    Sample 'shots' number of tensors from each tensor in train_features_unseen.
    Args:
        train_features_unseen (list): A list of tensors.
        shots (int): The number of samples to take from each tensor.
    Returns:
        list: A list of sampled tensors.
    """
    # Check if shots is not greater than the size of any tensor
    # min_size = min(tensor.size(0) for tensor in train_features_unseen)
    # if shots > min_size:
    #     raise ValueError("Shots cannot be greater than the size of any tensor.")
    # Sample shots number of elements from each tensor
    sampled_features = [tensor[torch.randperm(tensor.size(0))[:shots]] for tensor in train_features_unseen]
    return sampled_features

def run_regularized(K_list, alpha_list, V_final, train_features, test_features_sparse, 
                       train_features_unseen, test_features_sparse_unseen, N, N_unseen, device):
    """
    Compute accuracies for joint and few-shot learning.

    Parameters:
    K_list (list): List of values for K.
    alpha_list (list): List of values for alpha.
    V_final (tensor): Final value of V. 
    train_features (tensor): Training features.
    test_features_sparse (tensor): Test features for seen users.
    train_features_unseen (tensor): Training features for unseen users.
    test_features_sparse_unseen (tensor): Test features for unseen users.
    N (int): Number of seen users.
    N_unseen (int): Number of unseen users.
    device (device): Device to use for computations.

    Returns:
    tuple: Tuple containing 9 numpy arrays with computed accuracies and standard deviations.
    """

    # Initialize lists to store results
    train_accuracies_joint = []
    seen_user_unseen_prompts_accuracies_joint = []
    few_shot_train_accuracies_few_shot = []
    unseen_user_unseen_prompts_accuracies_few_shot = []
    train_accuracies_joint_std = []
    seen_user_unseen_prompts_accuracies_joint_std = []
    few_shot_train_accuracies_few_shot_std = []
    unseen_user_unseen_prompts_accuracies_few_shot_std = []

    for alpha in alpha_list:
        print("alpha : ", alpha)

        # Joint Reward and Weights Learning
        for K in K_list:
            print("Rank : ", K)
            if K == 0:
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
            else: 
                W_joint, V_joint = solve_regularized_simplex(V_final, alpha, train_features, K, num_iterations= 20000, learning_rate=0.5)
            
                # Save V_joint to file
                filename = f"/checkpoint/ai_society/representative_llms/data/lore/community/PRISM_V_lore_K_{K}_alpha_{alpha}.pt"
                torch.save(V_joint, filename)
                # Save Wsimulate_population_joint to file
                filename = f"/checkpoint/ai_society/representative_llms/data/lore/community/PRISM_W_lore_seen_{K}_{alpha}.pt"
                torch.save(W_joint.detach().cpu(), filename)

            print("Train Performance")
            accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)
            train_accuracies_joint.append(np.mean(accuracies_train))
            train_accuracies_joint_std.append(np.std(accuracies_train))

            print("Seen User Unseen Prompts")
            accuracies_seen_user_unseen_prompts = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], test_features_sparse)
            seen_user_unseen_prompts_accuracies_joint.append(np.mean(accuracies_seen_user_unseen_prompts))
            seen_user_unseen_prompts_accuracies_joint_std.append(np.std(accuracies_seen_user_unseen_prompts))

            # Learn the w on unseen users with few shot interactions
            if K <= 1:
                W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
            else:
                W_few_shot = learn_multiple_few_shot(train_features_unseen, V_joint.detach(), num_iterations=500, learning_rate=0.5)

            # Save W_joint to file
            # filename = f"checkpoints/W_lore_unseen_{K}.pt"
            # torch.save(torch.stack(W_few_shot).detach().cpu(), filename)

            print("Few Shot Train Performance")
            accuracies_few_shot_train = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen)
            few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
            few_shot_train_accuracies_few_shot_std.append(np.std(accuracies_few_shot_train))

            print("Unseen User Unseen Prompts")
            accuracies_unseen_user_unseen_prompts = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen)
            unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_user_unseen_prompts))
            unseen_user_unseen_prompts_accuracies_few_shot_std.append(np.std(accuracies_unseen_user_unseen_prompts))

    fac = 0.25
    train_accuracies_joint = np.array(train_accuracies_joint)
    seen_user_unseen_prompts_accuracies_joint = np.array(seen_user_unseen_prompts_accuracies_joint)
    few_shot_train_accuracies_few_shot = np.array(few_shot_train_accuracies_few_shot)
    unseen_user_unseen_prompts_accuracies_few_shot = np.array(unseen_user_unseen_prompts_accuracies_few_shot)
    train_accuracies_joint_std = fac * np.array(train_accuracies_joint_std)
    seen_user_unseen_prompts_accuracies_joint_std = fac * np.array(seen_user_unseen_prompts_accuracies_joint_std)
    few_shot_train_accuracies_few_shot_std = fac * np.array(few_shot_train_accuracies_few_shot_std)
    unseen_user_unseen_prompts_accuracies_few_shot_std = fac * np.array(unseen_user_unseen_prompts_accuracies_few_shot_std)

    return train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std




def run_few_shot_vary_shots(trials, alpha_list, K_list, num_shots, train_features, test_features, train_features_unseen, test_features_sparse_unseen, V_final, N, N_unseen, device):

    all_results = {}
    
    for alpha in alpha_list:
        print("alpha : ", alpha)
        
        # Joint Reward and Weights Learning
        accuracies_test_list =  []
        accuracies_train_list = []
        
        few_shot_train_accuracies_few_shot_means_list = []
        few_shot_train_accuracies_few_shot_stds_list = []
        unseen_user_unseen_prompts_accuracies_few_shot_means_list = []
        unseen_user_unseen_prompts_accuracies_few_shot_stds_list = []


        for K in K_list:
            print("K : ", K)
            if K == 0:
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
            else: 
                W_joint, V_joint = solve_regularized(V_final, alpha, train_features, K, num_iterations=500, learning_rate=0.5)
            
            print("Train Performance")
            accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)            
            accuracies_test = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], test_features)
            
            
            train_accuracies_joint = np.mean(accuracies_train)
            test_accuracies_joint = np.mean(accuracies_test)
            accuracies_train_list.append(train_accuracies_joint)
            accuracies_test_list.append(test_accuracies_joint)

            few_shot_train_accuracies_few_shot_means = []
            few_shot_train_accuracies_few_shot_stds = []
            unseen_user_unseen_prompts_accuracies_few_shot_means = []
            unseen_user_unseen_prompts_accuracies_few_shot_stds = []
            
            for shots in num_shots:
                print("Shots : ", shots)
                few_shot_train_accuracies_few_shot = []
                unseen_user_unseen_prompts_accuracies_few_shot = []
                
                for _ in range(trials):  # Run the experiment 10 times
                    # train_features_unseen = create_dataset_prism_shots(unseen_user_seen_dialog_embeddings, shots)
                    train_features_unseen_shots = sample_shots(train_features_unseen, shots)
                    #là on pourrait 
                    # Learn the w on unseen users with few shot interactions
                    if K <= 1:
                        W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
                    else:
                        W_few_shot = learn_multiple_few_shot(train_features_unseen_shots, V_joint.detach(), num_iterations=500, learning_rate=0.1)
                    
                    print("Few Shot Train Performance")
                    accuracies_few_shot_train = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen_shots)
                    few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
                    
                    print("Unseen User Unseen Prompts")
                    accuracies_unseen_user_unseen_prompts = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen)
                    unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_user_unseen_prompts))
                
                few_shot_train_accuracies_few_shot_means.append(np.mean(few_shot_train_accuracies_few_shot))
                few_shot_train_accuracies_few_shot_stds.append(np.std(few_shot_train_accuracies_few_shot))
                unseen_user_unseen_prompts_accuracies_few_shot_means.append(np.mean(unseen_user_unseen_prompts_accuracies_few_shot))
                unseen_user_unseen_prompts_accuracies_few_shot_stds.append(np.std(unseen_user_unseen_prompts_accuracies_few_shot))
            
            few_shot_train_accuracies_few_shot_means_list.append(few_shot_train_accuracies_few_shot_means)
            few_shot_train_accuracies_few_shot_stds_list.append(few_shot_train_accuracies_few_shot_stds)
            unseen_user_unseen_prompts_accuracies_few_shot_means_list.append(unseen_user_unseen_prompts_accuracies_few_shot_means)
            unseen_user_unseen_prompts_accuracies_few_shot_stds_list.append(unseen_user_unseen_prompts_accuracies_few_shot_stds)
          
    return few_shot_train_accuracies_few_shot_means_list, few_shot_train_accuracies_few_shot_stds_list, unseen_user_unseen_prompts_accuracies_few_shot_means_list, unseen_user_unseen_prompts_accuracies_few_shot_stds_list,  unseen_user_unseen_prompts_pareto_few_shot_means_list, unseen_user_unseen_prompts_pareto_few_shot_stds_list,accuracies_train_list, accuracies_test_list



##################
################# NEW: putting somewhere else? 

def run_few_shot_vary_shots_align(trials, alpha, K_list, num_shots, train_features, test_features, train_features_unseen, test_features_sparse_unseen, alignment_param,V_final, N,N_unseen, device):

    #unseen user preference accuracies (test/train)
    unseen_user_test_prompts_preference_accuracies_means = []
    unseen_user_test_prompts_preference_accuracies_std = []
    unseen_user_train_prompts_preference_accuracies_means = []
    unseen_user_train_prompts_preference_accuracies_std = []

    #seen user preference accuracies (test/train)
    seen_user_test_prompts_preference_accuracies = []
    seen_user_train_prompts_preference_accuracies = []

    #unseen user winrates 
    unseen_user_unseen_prompts_winrate_means = []
    unseen_user_unseen_prompts_winrate_stds  = []

    #other
    unseen_user_unseen_prompts_pareto_means = []
    unseen_user_unseen_prompts_pareto_stds = []

    for K in K_list:
        print("K : ", K)
        if K == 0:
            V_joint = V_final
            W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
        else: 
            W_joint, V_joint = solve_regularized(V_final, alpha, train_features, K, num_iterations=500, learning_rate=0.5)
        
        print("Train Performance")
        accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)            
        accuracies_test = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], test_features)
        
        
        train_accuracies_joint = np.mean(accuracies_train)
        test_accuracies_joint = np.mean(accuracies_test)

        seen_user_train_prompts_preference_accuracies.append(train_accuracies_joint)
        seen_user_test_prompts_preference_accuracies.append(test_accuracies_joint)

        #unseen user preference accuracies (test/train)
        unseen_user_test_prompts_preference_accuracies_means_K = []
        unseen_user_test_prompts_preference_accuracies_std_K = []
        unseen_user_train_prompts_preference_accuracies_means_K = []
        unseen_user_train_prompts_preference_accuracies_std_K = []

        #unseen user winrates 
        unseen_user_unseen_prompts_winrate_means_K = []
        unseen_user_unseen_prompts_winrate_stds_K  = []

        #other
        unseen_user_unseen_prompts_pareto_means_K = []
        unseen_user_unseen_prompts_pareto_stds_K = []         

        for shots in num_shots:
            print("Shots : ", shots)
            #unseen user preference accuracies (test/train)
            unseen_user_test_prompts_preference_accuracies_tmp = []
            unseen_user_train_prompts_preference_accuracies_tmp = []

            #unseen user winrates 
            unseen_user_unseen_prompts_winrate_tmp = []

            #other
            unseen_user_unseen_prompts_pareto_tmp = []

            for _ in range(trials):  # Run the experiment 10 times
                # train_features_unseen = create_dataset_prism_shots(unseen_user_seen_dialog_embeddings, shots)
                train_features_unseen_shots = sample_shots(train_features_unseen, shots)
                # Learn the w on unseen users with few shot interactions
                if K <= 1:
                    W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
                else:
                    W_few_shot = learn_multiple_few_shot(train_features_unseen_shots, V_joint.detach(), num_iterations=500, learning_rate=0.1)
                
                unseen_user_train_prompts_preference_accuracies = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen_shots)
                unseen_user_train_prompts_preference_accuracies_tmp.append(np.mean(unseen_user_train_prompts_preference_accuracies))
                
                #PL accuracy
                unseen_user_test_prompts_preference_accuracies = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen)
                unseen_user_test_prompts_preference_accuracies_tmp.append(np.mean(unseen_user_test_prompts_preference_accuracies))
                
                #PA accuracy 
                unseen_user_unseen_prompts_winrate, unseen_user_unseen_prompts_pareto = eval_multiple_pa(W_few_shot, [V_joint.detach() for i in range(N_unseen)], alignment_param) #np.zeros( N_unseen), np.zeros( N_unseen) 
                unseen_user_unseen_prompts_pareto_tmp.append(np.mean(unseen_user_unseen_prompts_pareto))
                unseen_user_unseen_prompts_winrate_tmp.append(np.mean(unseen_user_unseen_prompts_winrate))


            unseen_user_test_prompts_preference_accuracies_means_K.append(np.mean(unseen_user_test_prompts_preference_accuracies_tmp))
            unseen_user_test_prompts_preference_accuracies_std_K.append(np.std(unseen_user_test_prompts_preference_accuracies_tmp))
            unseen_user_train_prompts_preference_accuracies_means_K.append(np.mean(unseen_user_train_prompts_preference_accuracies_tmp))
            unseen_user_train_prompts_preference_accuracies_std_K.append(np.std(unseen_user_train_prompts_preference_accuracies_tmp))
            unseen_user_unseen_prompts_winrate_means_K.append(np.mean(unseen_user_unseen_prompts_winrate_tmp))
            unseen_user_unseen_prompts_winrate_stds_K.append(np.std(unseen_user_unseen_prompts_winrate_tmp))    
            unseen_user_unseen_prompts_pareto_means_K.append(np.mean(unseen_user_unseen_prompts_pareto_tmp))
            unseen_user_unseen_prompts_pareto_stds_K.append(np.std(unseen_user_unseen_prompts_pareto_tmp))


        unseen_user_test_prompts_preference_accuracies_means.append(unseen_user_test_prompts_preference_accuracies_means_K)
        unseen_user_test_prompts_preference_accuracies_std.append(unseen_user_test_prompts_preference_accuracies_std_K)
        unseen_user_train_prompts_preference_accuracies_means.append(unseen_user_train_prompts_preference_accuracies_means_K)
        unseen_user_train_prompts_preference_accuracies_std.append(unseen_user_train_prompts_preference_accuracies_std_K)
        unseen_user_unseen_prompts_winrate_means.append(unseen_user_unseen_prompts_winrate_means_K)
        unseen_user_unseen_prompts_winrate_stds.append(unseen_user_unseen_prompts_winrate_stds_K)    
        unseen_user_unseen_prompts_pareto_means.append(unseen_user_unseen_prompts_pareto_means_K)
        unseen_user_unseen_prompts_pareto_stds.append(unseen_user_unseen_prompts_pareto_stds_K)

    return {
        'unseen_user_test_prompts_preference_accuracies_means': unseen_user_test_prompts_preference_accuracies_means,
        'unseen_user_test_prompts_preference_accuracies_std': unseen_user_test_prompts_preference_accuracies_std,
        'unseen_user_train_prompts_preference_accuracies_means': unseen_user_train_prompts_preference_accuracies_means,
        'unseen_user_train_prompts_preference_accuracies_std': unseen_user_train_prompts_preference_accuracies_std,
        'seen_user_test_prompts_preference_accuracies': seen_user_test_prompts_preference_accuracies,
        'seen_user_train_prompts_preference_accuracies': seen_user_train_prompts_preference_accuracies,
        'unseen_user_unseen_prompts_winrate_means': unseen_user_unseen_prompts_winrate_means,
        'unseen_user_unseen_prompts_winrate_stds': unseen_user_unseen_prompts_winrate_stds,
        'unseen_user_unseen_prompts_pareto_means': unseen_user_unseen_prompts_pareto_means,
        'unseen_user_unseen_prompts_pareto_stds': unseen_user_unseen_prompts_pareto_stds
    }


def eval_multiple_pa(W_list, V_list, alignment_param):
    accuracies = []
    N = len(V_list) # number of unseen users
    results = [evaluate_model_pa( V_list[i], W_list[i], alignment_param, i) for i in range(N)]
    print(f'results: {results}')
    accuracies = [r[0] for r in results]
    pareto_results = [r[1] for r in results]
    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(accuracies)
    print(f"Average accuracy over {N} unseen users: {average_accuracy:.4f}")
    print(f"Standard deviation of accuracy: {std_accuracy:.4f}")
    print(f"Average Pareto accuracy over {N} unseen users: {np.mean(pareto_results):.4f}")
    print(f"Standard deviation of Pareto accuracy: {np.std(pareto_results):.4f}")
    return accuracies, pareto_results

def evaluate_model_pa( V, W, alignment_param, user_index):

    #extract all user descriptions
    user_descriptions_unseen= alignment_param["user_descriptions_unseen"]
    # unseen_user_id
    unseen_user_id = alignment_param["unseen_user_id"]

    user_row = user_descriptions_unseen[user_descriptions_unseen['persona_uuid'] == unseen_user_id[user_index]]
    user_persona = user_row['score_persona'].iloc[0]  # Get the dict
    user_description = user_persona['persona_description']  # Get nested value    

    #extract user test prompts, answers and embeddings
    prompts_unseen_test = alignment_param["prompts_unseen_test"]
    bestofn_answers_embeddings_prompts_unseen_test= alignment_param["bestofn_answers_embeddings_prompts_unseen_test"]
    bestofn_answers_prompts_unseen_test= alignment_param["bestofn_answers_prompts_unseen_test"]
    
    
    prompts_user = prompts_unseen_test[user_index]
    answers_user = bestofn_answers_prompts_unseen_test[user_index]
    answers_embeddings_user = bestofn_answers_embeddings_prompts_unseen_test[user_index]

    #judge model 
    judge_model= alignment_param["judge_model"]
    judge_sampling_params = alignment_param["judge_sampling_params"]


    """
    Evaluate the model's performance on a test set (of a specific user) using a judge model.

    Parameters:
    V (torch.Tensor): The reward model parameters.
    W (torch.Tensor): The weight vector for the user.
    alignment_param (dict): A dictionary containing all alignment parameters, including:
        - test_instruction_sparse_unseen (torch.Tensor): The test instructions for unseen prompts.
        - user_descriptions_unseen (str): The descriptions of the unseen users.
        - base_model: The base language model to generate responses.
        - base_tokenizer: The tokenizer for the base language model.
        - judge_model: The language model used as a judge to evaluate responses.
        - judge_sampling_params: The sampling parameters for the judge model.

    Returns:
    float: The accuracy of the model's responses as judged by the judge model.
    """


    accuracies = []
    pareto_dominances = []
    # Generate responses using the base model and evaluate with the judge model
    # This is a placeholder implementation and should be replaced with actual logic
    instructions = []
    personalized_indices = []
    for prompt_idx in range(len(prompts_user)):
        prompt = prompts_user[prompt_idx]
        generated_responses = answers_user[prompt_idx]
        answers_embeddings = answers_embeddings_user[prompt_idx]
        #print('prompt : ', prompt)
        #generated_responses = generate_responses(base_model, base_tokenizer, prompt, N_bestofN)
        first_answer = generated_responses[0]  # Assuming the first response is the one we want to evaluate
        #print('base model generation :')
        #print ( first_answer)
        best_of_n_index,pareto_dominates_first_answer = compute_best_of_n_responses( V,W, answers_embeddings )  # Implement this function to select the best response based on the judge model
        best_of_n_responses = generated_responses[best_of_n_index]
        instruction, personalized_index = create_instruction(prompt, user_description, first_answer, best_of_n_responses)
        instructions.append(instruction)
        personalized_indices.append(personalized_index)
        pareto_dominances.append(pareto_dominates_first_answer)
        
    accuracies = evaluate_with_judge(instructions, judge_model, judge_sampling_params,personalized_indices)
    user_accuracy = np.mean(accuracies)
    percentage_judge_dectect_pareto_dominance = np.sum((pareto_dominances == accuracies) & (pareto_dominances == 1))/np.sum(pareto_dominances)
    print('user accuracy : ', user_accuracy)
    return [user_accuracy, percentage_judge_dectect_pareto_dominance]

def compute_best_of_n_responses( V, W, embeddings):

      # Stack embeddings (list of 1D tensors) into a 2D tensor [N, 4096]
      embeddings_tensor = embeddings.float()  # Ensure float32 after stacking
      
      # Ensure V and W are on the same device as embeddings and in float32
      device = embeddings_tensor.device
      #print(f'device embeddings {device}')
      V = V.to(device).float()  # .float() ensures float32
      W = W.to(device).float()  # .float() ensures float32
      
      #print(f"DEBUG: embeddings_tensor dtype={embeddings_tensor.dtype}, V dtype={V.dtype}, W dtype={W.dtype}")
      criteria = embeddings_tensor @ V
      rewards = criteria @ W
      #print(rewards)
      best_index = torch.argmax(rewards).item()
      pareto_dominates_first_answer = (criteria[best_index,:] >= criteria[0,:]).sum().item() == criteria.shape[1]
      #print(f"Best response index: {best_index}, Pareto dominates first answer: {pareto_dominates_first_answer}")
      return best_index, pareto_dominates_first_answer

def create_instruction(prompt, user_description, first_answer, best_of_n_responses):
    text_template_1 = f'You are tasked with comparing two answers to the question: {prompt} from the perspective of the following persona: {user_description}.'
    coin = np.random.rand()
    if coin < 0.5:
        text_template_2 = f'Answer A: {first_answer} \n\n Answer B: {best_of_n_responses} \n\n'
    else:
        text_template_2 = f'Answer A: {best_of_n_responses} \n\n Answer B: {first_answer} \n\n'
    
    #text_template_3 = f'Which answer does the persona prefer? Answer A, B, or Neither.\nRespond using this exact structure:\nAnswer: [A/B/Neither]\nReason: [brief explanation]\n\n'    
    text_template_3 = f'Which answer does the persona prefer? Answer only A, B, or Neither using this exact structure:\nAnswer: [A/B/Neither] \n\n'    
    
    instruction = text_template_1 + "\n\n" + text_template_2 + "\n\n" + text_template_3
    
    return instruction, coin < 0.5

def evaluate_with_judge(instructions, judge_model, judge_sampling_params, personalized_indices):
    """
    Evaluate the generated responses using a judge model.

    Parameters:
    judge_model: The language model used as a judge to evaluate responses.
    first_answer: The first generated response to be evaluated.
    best_of_n_responses: The best response selected from the N generated responses.

    Returns:
    float: The accuracy of the model's responses as judged by the judge model.
    """

    # generation with Transformer
    # # Set pad token if not set
    # if judge_tokenizer.pad_token is None:
    #     judge_tokenizer.pad_token = judge_tokenizer.eos_token
    
    # inputs = judge_tokenizer(instruction, return_tensors="pt", padding=True).to(model_device)
    
    # with torch.no_grad():
    #     answer_ids = judge_model.generate(
    #         inputs.input_ids, 
    #         attention_mask=inputs.get("attention_mask"),
    #         max_new_tokens=100, 
    #         do_sample=True, 
    #         top_p=0.9,
    #         pad_token_id=judge_tokenizer.pad_token_id
    #     )
    
    # Decode only the newly generated tokens (not the input prompt)
    # input_length = inputs.input_ids.shape[1]
    # generated_ids = answer_ids[0, input_length:]
    # answer_text = judge_tokenizer.decode(generated_ids, skip_special_tokens=True)
    time_start = time.time()
    outputs = judge_model.generate(instructions, judge_sampling_params)
    time_end = time.time()
    print(f"Judge model generation time: {time_end - time_start:.2f} seconds")
    accuracies = []

    for i, output in enumerate(outputs):
        # Safe extraction
        if not output.outputs:
            answer_text = ""
        else:
            answer_text = output.outputs[0].text.strip()

        # Normalize (important for robustness)
        first_line = answer_text.split("\n")[0].strip()

        # Extract prediction (robust parsing)
        if "A" in first_line:
            pref = "A"
        elif "B" in first_line:
            pref = "B"
        elif "Neither" in first_line:
            pref = "Neither"
        else:
            pref = None  # fallback

        # Ground truth
        personalized = "B" if personalized_indices[i] == 1 else "A"

        # Accuracy
        is_correct = (pref == personalized)
        accuracies.append(is_correct)

        if i == 0:
            print(instructions[i])    
            print(f"judge_answer: {answer_text}")
            print(f"predicted: {pref} | expected: {personalized}")
            print(is_correct)

    return  accuracies

          



