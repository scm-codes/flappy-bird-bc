"""
Model Evaluation and Leaderboard Generation

This module evaluates trained TorchScript models on the Flappy Bird game
and generates a leaderboard ranking based on performance metrics.

The evaluation process:
1. Loads TorchScript models from a specified directory
2. Runs each model for a fixed number of episodes with deterministic seeds
3. Calculates performance statistics (mean, std, best score)
4. Ranks models and saves results to CSV

Usage:
    python eval_leaderboard.py --folder models --episodes 25
"""

import argparse
import importlib
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class PolicyController:
    """
    TorchScript policy wrapper implementing the controller API expected by flappy.py.
    
    This class wraps a trained TorchScript model and provides the interface
    needed to control the Flappy Bird game. It converts game observations
    to model inputs and model outputs to game actions.
    """
    
    def __init__(self, jit_model: torch.jit.ScriptModule):
        """
        Initialize the policy controller.
        
        Args:
            jit_model: Trained TorchScript model for action prediction
        """
        self.model = jit_model.eval()
        self.auto_start = True  # Allow flappy.py to auto-tap on splash if needed

    def decide(self, obs: dict) -> int:
        """
        Make a decision based on the current game observation.
        
        Args:
            obs: Dictionary containing game state information
            
        Returns:
            Action to take (0 = no flap, 1 = flap)
        """
        # Convert observation to tensor in the same order as training data
        # Order matches flappy._il_make_obs
        obs_tensor = torch.tensor([
            obs["bird_y"],
            obs["bird_v"],
            obs["next_pipe_x"],
            obs["next_pipe_top"],
            obs["next_pipe_bottom"],
            obs["dist_to_pipe_center"],
        ], dtype=torch.float32).unsqueeze(0)
        
        with torch.inference_mode():
            logit = self.model(obs_tensor)  # shape [1]
            return int((logit.sigmoid() > 0.5).item())


def evaluate_model(
    model_path: Path,
    episodes: int,
    base_seed: int,
    headless: bool,
    turbo_steps: int,
    mute: bool,
) -> Tuple[float, float, int]:
    """
    Evaluate a single TorchScript model for the specified number of episodes.
    
    Args:
        model_path: Path to the TorchScript model file
        episodes: Number of episodes to run
        base_seed: Base random seed for deterministic evaluation
        headless: Whether to run without rendering
        turbo_steps: Physics steps per frame for speedup
        mute: Whether to disable audio
        
    Returns:
        Tuple of (mean_score, std_score, best_score)
    """
    # Lazy import so flappy.py picks up any globals (HEADLESS, etc.) we set
    flappy_bird = importlib.import_module("flappy_bird")

    # Configure evaluation settings for speed/IO
    flappy_bird.set_eval_options(headless=headless, turbo_steps=turbo_steps, mute=mute)

    # Load and wrap the policy model
    policy = torch.jit.load(str(model_path), map_location="cpu")
    controller = PolicyController(policy)
    flappy_bird.set_controller(controller)

    scores: list[int] = []

    # Make the environment stochastic but FAIR across models:
    # reseed Python's RNG before each episode, so every model sees the same sequence
    for episode in range(episodes):
        random.seed(base_seed + episode)
        try:
            score = flappy_bird.run_one_episode()  # Uses current eval options
        except Exception as e:
            # If a model crashes mid-episode, treat it as 0 for robustness
            print(f"[WARN] Episode {episode + 1} for {model_path.name} failed: {e}")
            score = 0
        scores.append(int(score))

    # Calculate statistics
    scores_array = np.asarray(scores, dtype=np.int32)
    mean_score = float(scores_array.mean())
    std_score = float(scores_array.std(ddof=0))
    best_score = int(scores_array.max(initial=0))
    
    return mean_score, std_score, best_score


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate TorchScript agents and build a leaderboard"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="models",
        help="Folder containing *.pt model files (default: models)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help="Number of episodes per model (default: 25)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="leaderboard.csv",
        help="Output CSV file path (default: leaderboard.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base RNG seed for deterministic evaluation (default: 1337)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run evaluation without rendering (faster)"
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Disable audio during evaluation"
    )
    parser.add_argument(
        "--turbo_steps",
        type=int,
        default=16,
        help="Physics steps per frame for speedup (default: 16)"
    )
    
    args = parser.parse_args()

    # Find all model files
    folder = Path(args.folder)
    model_paths = sorted(folder.glob("*.pt"))
    
    if not model_paths:
        print(f"No .pt models found in: {folder.resolve()}")
        return

    print(f"Evaluating {len(model_paths)} model(s) × {args.episodes} episode(s) ...")
    print(f"Options: headless={args.headless} turbo_steps={args.turbo_steps} "
          f"mute={args.mute} seed={args.seed}")

    # Evaluate each model
    results = []
    for model_path in model_paths:
        print(f"\n-> {model_path.name}")
        
        mean_score, std_score, best_score = evaluate_model(
            model_path=model_path,
            episodes=args.episodes,
            base_seed=args.seed,
            headless=args.headless,
            turbo_steps=max(1, args.turbo_steps),
            mute=args.mute,
        )
        
        results.append((model_path.name, mean_score, std_score, best_score))
        print(f"   mean={mean_score:.2f}  std={std_score:.2f}  best={best_score}")

    # Sort results: higher mean first, then lower std, then higher best
    results.sort(key=lambda r: (-r[1], r[2], -r[3]))

    # Print leaderboard
    print("\n===== Leaderboard =====")
    for rank, (name, mean_score, std_score, best_score) in enumerate(results, 1):
        print(f"{rank:2d}. {name:30s}  mean={mean_score:.2f}  "
              f"std={std_score:.2f}  best={best_score}")

    # Save results to CSV
    output_path = Path(args.csv)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("rank,name,mean,std,best\n")
        for rank, (name, mean_score, std_score, best_score) in enumerate(results, 1):
            f.write(f"{rank},{name},{mean_score:.6f},{std_score:.6f},{best_score}\n")
    
    print(f"\nSaved leaderboard to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
