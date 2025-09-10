"""
Imitation Learning Data Collection Wrapper

This module provides functionality to record human gameplay data for behavior cloning.
It implements a HumanRecorder class that captures observations and actions during
human gameplay sessions and saves them to JSONL files for training.

Usage:
    python il_wrapper.py --record --episodes 10
"""

import argparse
import importlib
import json
import time
from pathlib import Path


# Directory structure for storing recorded sessions
DATA_DIR = Path("il_data")
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class HumanRecorder:
    """
    Records human gameplay data for imitation learning.
    
    This class implements the controller interface expected by flappy.py
    and captures observation-action pairs during human gameplay sessions.
    Data is saved to JSONL files with timestamps and episode rewards.
    """
    
    def __init__(self):
        """Initialize the recorder with no active session."""
        self._cur_file = None
        self._cur_path = None

    def start_episode(self):
        """
        Start recording a new episode.
        
        Creates a new JSONL file with a timestamp-based name to store
        observation-action pairs for this episode.
        """
        timestamp = int(time.time() * 1000)
        name = f"session_{timestamp}.jsonl"
        self._cur_path = SESSIONS_DIR / name
        self._cur_file = open(self._cur_path, "w", encoding="utf-8")

    def end_episode(self, reward: int = None):
        """
        End the current episode and optionally rename the file with reward.
        
        Args:
            reward: The final score/reward for this episode. If provided,
                   the session file will be renamed to include this reward.
        """
        if self._cur_file:
            self._cur_file.close()
            self._cur_file = None
            
            if reward is not None:
                # Rename file to include reward for easy identification
                new_name = f"{self._cur_path.stem}_R{reward}{self._cur_path.suffix}"
                new_path = self._cur_path.with_name(new_name)
                self._cur_path.rename(new_path)
                print(f"Saved: {new_path}")

    def decide(self, obs):
        """
        Controller decision method - returns None to allow human input.
        
        Args:
            obs: Current game observation dictionary
            
        Returns:
            None: Always returns None to not override human input
        """
        return None  # Don't override human input

    def record(self, obs, action):
        """
        Record an observation-action pair to the current session file.
        
        Args:
            obs: Observation dictionary from the game
            action: Action taken (0 or 1 for flap/no-flap)
        """
        if self._cur_file:
            data = {"obs": obs, "action": int(action)}
            self._cur_file.write(json.dumps(data) + "\n")


def run_record(episodes: int = 10):
    """
    Run data collection sessions for the specified number of episodes.
    
    Args:
        episodes: Number of episodes to record (default: 10)
    """
    # Import the flappy game module
    flappy_bird = importlib.import_module("flappy_bird")
    
    # Create and attach the recorder
    recorder = HumanRecorder()
    flappy_bird.set_controller(recorder)

    print(f"Play {episodes} episodes. SPACE/UP to flap, ESC closes a window.")
    
    for episode in range(episodes):
        # Run one episode - the recorder will automatically start/end
        # and capture all observation-action pairs
        score = flappy_bird.run_one_episode()
        print(f"[episode {episode + 1}/{episodes}] score={score}")


def main():
    """Main entry point for the data collection script."""
    parser = argparse.ArgumentParser(
        description="Record human gameplay data for behavior cloning"
    )
    parser.add_argument(
        "--record", 
        action="store_true",
        help="Start recording human gameplay sessions"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=10,
        help="Number of episodes to record (default: 10)"
    )
    
    args = parser.parse_args()
    
    if args.record:
        run_record(args.episodes)
    else:
        print("Use: python il_wrapper.py --record --episodes 10")


if __name__ == "__main__":
    main()
