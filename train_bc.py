"""
Behavior Cloning Training Script

This module implements vanilla behavior cloning for the Flappy Bird game.
It processes recorded human gameplay data and trains a neural network to
imitate human behavior using supervised learning.

The training process consists of two main steps:
1. Merge recorded session files into a single dataset
2. Train a multi-layer perceptron (MLP) to predict human actions

Usage:
    python train_bc.py --merge  # Process recorded data
    python train_bc.py --train --model_id "your_name"  # Train model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# Directory structure and file paths
DATA_DIR = Path("il_data")
SESSIONS_DIR = DATA_DIR / "sessions"
MERGED_DATASET = DATA_DIR / "dataset_merged.npz"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for behavior cloning.
    
    This neural network takes game observations as input and outputs
    a binary logit for the flap action (1) vs no action (0).
    
    Architecture:
        - Layer normalization for input stability
        - Two hidden layers with ReLU activation
        - Single output neuron with sigmoid activation
    """
    
    def __init__(self, obs_dim: int = 6, hidden: int = 64):
        """
        Initialize the MLP model.
        
        Args:
            obs_dim: Number of observation features (default: 6)
            hidden: Number of neurons in hidden layers (default: 64)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, obs_dim]
            
        Returns:
            Output logits of shape [batch_size]
        """
        return self.net(x).squeeze(-1)


def merge_sessions():
    """
    Merge all recorded session files into a single dataset.
    
    Reads all session_*.jsonl files from the sessions directory and
    combines them into a single compressed numpy file for training.
    The dataset contains observation vectors and corresponding actions.
    """
    observations = []
    actions = []
    
    # Process all session files
    session_files = sorted(SESSIONS_DIR.glob("session_*.jsonl"))
    if not session_files:
        print(f"No session files found in {SESSIONS_DIR}")
        return
    
    for session_file in session_files:
        print(f"Processing {session_file.name}...")
        
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                obs = data["obs"]
                action = int(data["action"])
                
                # Extract observation features in the expected order
                obs_vector = [
                    obs["bird_y"],
                    obs["bird_v"],
                    obs["next_pipe_x"],
                    obs["next_pipe_top"],
                    obs["next_pipe_bottom"],
                    obs["dist_to_pipe_center"]
                ]
                
                observations.append(obs_vector)
                actions.append(action)
    
    # Convert to numpy arrays
    X = np.asarray(observations, dtype=np.float32)
    y = np.asarray(actions, dtype=np.int64)
    
    # Save merged dataset
    np.savez_compressed(MERGED_DATASET, X=X, y=y)
    print(f"Merged {len(y)} samples -> {MERGED_DATASET}")


def train_model(
    model_id: str = "anon",
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 64
):
    """
    Train a behavior cloning model on the merged dataset.
    
    Args:
        model_id: Identifier for the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for AdamW optimizer
        hidden_size: Number of neurons in hidden layers
    """
    # Load the merged dataset
    try:
        data = np.load(MERGED_DATASET)
        X = torch.from_numpy(data["X"])
        y = torch.from_numpy(data["y"])
    except FileNotFoundError:
        print(f"Dataset not found: {MERGED_DATASET}")
        print("Run 'python train_bc.py --merge' first to create the dataset.")
        return
    
    print(f"Training on {len(y)} samples...")
    
    # Handle class imbalance with weighted sampling
    class_counts = torch.bincount(y, minlength=2).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y]
    
    # Create data loader with weighted sampling
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(y), 
        replacement=True
    )
    dataloader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        sampler=sampler
    )
    
    # Initialize model and training components
    model = MLP(obs_dim=X.shape[1], hidden=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_x, batch_y in dataloader:
            # Forward pass
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y.float())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item() * batch_x.size(0)
            predictions = (logits.sigmoid() > 0.5).long()
            correct_predictions += (predictions == batch_y).sum().item()
            total_samples += batch_x.size(0)
        
        # Print epoch statistics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={accuracy:.3f}")
    
    # Save the trained model as TorchScript
    model.eval()
    scripted_model = torch.jit.script(model)
    output_path = MODELS_DIR / f"{model_id}.pt"
    scripted_model.save(str(output_path))
    print(f"Saved: {output_path}")
    print("Submit ONLY this .pt file.")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train behavior cloning models for Flappy Bird"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge recorded session files into training dataset"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a behavior cloning model"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="anon",
        help="Student identifier for the model (default: anon)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=256,
        help="Batch size for training (default: 256)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Hidden layer size (default: 64)"
    )
    
    args = parser.parse_args()
    
    if args.merge:
        merge_sessions()
    
    if args.train:
        train_model(
            model_id=args.model_id,
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            hidden_size=args.hidden
        )


if __name__ == "__main__":
    main()
