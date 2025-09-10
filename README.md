# Flappy Bird Behavior Cloning

A Flappy Bird game implementation in Python with behavior cloning (imitation learning) capabilities. This project allows you to record human gameplay data and train neural networks to imitate human behavior.

## Overview

This repository contains a complete behavior cloning pipeline for the Flappy Bird game:

1. **Game Engine** (`flappy_bird.py`) - The core Flappy Bird game with imitation learning hooks
2. **Data Collection** (`il_wrapper.py`) - Record human gameplay sessions
3. **Model Training** (`train_bc.py`) - Train neural networks using supervised learning
4. **Model Evaluation** (`eval_leaderboard.py`) - Evaluate and rank trained models

## Features

- **Human Data Collection**: Record observation-action pairs during gameplay
- **Behavior Cloning**: Train MLPs to imitate human behavior
- **Model Evaluation**: Comprehensive evaluation with leaderboards
- **TorchScript Support**: Export trained models for efficient inference
- **Deterministic Evaluation**: Fair comparison across different models

## Dependencies

```bash
pip install pygame torch numpy
```

Required packages:
- `pygame` - Game engine and graphics
- `torch` - Neural network training and inference
- `numpy` - Numerical computations

## Quick Start

### 1. Record Human Gameplay Data

First, record some human gameplay sessions to create training data:

```bash
python il_wrapper.py --record --episodes 10
```

This will:
- Launch the Flappy Bird game
- Record 10 episodes of your gameplay
- Save observation-action pairs to `il_data/sessions/`

**Controls:**
- `SPACE` or `UP` arrow - Make the bird flap
- `ESC` - Close the game window

### 2. Process Training Data

Merge all recorded sessions into a single dataset:

```bash
python train_bc.py --merge
```

This creates `il_data/dataset_merged.npz` containing all training data.

### 3. Train a Behavior Cloning Model

Train a neural network to imitate your gameplay:

```bash
python train_bc.py --train --model_id "your_name" --epochs 20
```

This will:
- Load the merged dataset
- Train an MLP with weighted sampling for class balance
- Save the trained model as `models/your_name.pt`

### 4. Evaluate Your Model

Test your trained model and generate a leaderboard:

```bash
python eval_leaderboard.py --folder models --episodes 25 --headless
```

The `--headless` flag runs evaluation without rendering for faster results.

## Detailed Usage

### Data Collection (`il_wrapper.py`)

```bash
# Record 20 episodes with custom settings
python il_wrapper.py --record --episodes 20
```

**Output:** Session files in `il_data/sessions/session_*.jsonl`

### Model Training (`train_bc.py`)

```bash
# Basic training with default parameters
python train_bc.py --merge --train --student_id "my_model"

# Advanced training with custom hyperparameters
python train_bc.py --train --student_id "advanced_model" \
    --epochs 50 --batch 512 --lr 0.001 --hidden 128
```

**Parameters:**
- `--merge`: Process recorded sessions into training dataset
- `--train`: Train a behavior cloning model
- `--student_id`: Model identifier (used for output filename)
- `--epochs`: Number of training epochs (default: 10)
- `--batch`: Batch size (default: 256)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden`: Hidden layer size (default: 64)

**Output:** Trained model in `models/student_id.pt`

### Model Evaluation (`eval_leaderboard.py`)

```bash
# Evaluate models in models folder
python eval_leaderboard.py --folder models --episodes 50

# Fast evaluation without rendering
python eval_leaderboard.py --folder models --episodes 25 --headless --turbo_steps 32

# Custom evaluation settings
python eval_leaderboard.py --folder models --episodes 100 \
    --seed 42 --csv "my_leaderboard.csv" --headless --mute
```

**Parameters:**
- `--folder`: Directory containing `.pt` model files
- `--episodes`: Episodes per model (default: 25)
- `--csv`: Output CSV file (default: leaderboard.csv)
- `--seed`: Random seed for deterministic evaluation (default: 1337)
- `--headless`: Run without rendering (faster)
- `--mute`: Disable audio
- `--turbo_steps`: Physics steps per frame for speedup (default: 16)

**Output:** 
- Console leaderboard with rankings
- CSV file with detailed results

## Model Architecture

The behavior cloning model is a Multi-Layer Perceptron (MLP) with:

- **Input**: 6-dimensional observation vector
  - Bird Y position
  - Bird velocity
  - Next pipe X position
  - Next pipe top Y position
  - Next pipe bottom Y position
  - Distance to pipe center

- **Architecture**:
  - Layer normalization
  - Two hidden layers with ReLU activation
  - Single output neuron (binary classification)

- **Training**: Binary cross-entropy loss with weighted sampling for class balance

## File Structure

```
flappy-bird-python/
├── flappy_bird.py         # Core game engine with IL hooks
├── il_wrapper.py          # Human data collection
├── train_bc.py            # Model training pipeline
├── eval_leaderboard.py    # Model evaluation and ranking
├── assets/                # Game sprites and audio
├── il_data/               # Recorded gameplay data
│   ├── sessions/          # Individual session files
│   └── dataset_merged.npz # Merged training dataset
├── models/                # Trained model files
```

## Tips for Better Performance

1. **Data Quality**: Record diverse gameplay sessions with varying strategies
2. **Data Quantity**: More training data generally leads to better performance
3. **Hyperparameters**: Experiment with learning rates, batch sizes, and model sizes
4. **Evaluation**: Use deterministic seeds for fair model comparison
5. **Speed**: Use `--headless` and `--turbo_steps` for faster evaluation

## Credits

- Game engine references
  - [repo](https://github.com/LeonMarqs/Flappy-bird-python), 
  - [video](https://www.youtube.com/watch?v=UZg49z76cLw&list=PL8ui5HK3oSiF7ZFfwYokCD5myWYhGH24A)
- ChatGPT for code assist, documentation and PEP 8 conventions 
- Original Flappy Bird game by Dong Nguyen

## License

This project is licensed under the MIT License - see the LICENSE file for details.



