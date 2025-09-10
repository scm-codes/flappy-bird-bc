"""
Flappy Bird Game Engine with Imitation Learning Support

This module implements the core Flappy Bird game with hooks for imitation learning
and model evaluation. It provides a complete game environment that can be controlled
by human players or AI agents through a standardized controller interface.

The game supports:
- Human gameplay with keyboard controls
- AI agent control through external controllers
- Data collection for behavior cloning
- Fast evaluation modes for model testing
- Deterministic gameplay for fair comparisons

Usage:
    python flappy_bird.py  # Play a single episode
"""

import random
import time
from typing import Optional, Tuple

import pygame
from pygame.locals import K_SPACE, K_UP, QUIT


# ===================== IMITATION LEARNING & EVALUATION HOOKS =====================

# Global controller for external AI agents
EXTERNAL_CONTROLLER: Optional[object] = None

# Last episode's score (used by evaluators)
LAST_SCORE: int = 0

# Auto-start delay for AI agents (frames to wait before auto-tapping splash screen)
AUTO_START_DELAY_FRAMES: int = 6  # ~0.4s at 15 FPS

# Evaluation mode toggles (defaults are human-friendly)
HEADLESS: bool = False      # Hide window and skip rendering when True
TURBO_STEPS: int = 1        # Physics steps per frame; >1 speeds up evaluation
MUTE: bool = False          # Disable audio playback


def set_controller(controller: object) -> None:
    """
    Attach an external controller for AI agent control.
    
    The controller can implement the following optional methods:
    - start_episode(): Called at the beginning of each episode
    - end_episode(score): Called at the end of each episode with final score
    - decide(obs: dict) -> int: Return 0/1 to override human input, None to allow human control
    - record(obs: dict, action: int): Called every frame to record observation-action pairs
    
    Optional attributes:
    - auto_start: bool (default True) - Allow auto-tap on splash screen if agent is idle
    
    Args:
        controller: Object implementing the controller interface
    """
    global EXTERNAL_CONTROLLER
    EXTERNAL_CONTROLLER = controller


def set_eval_options(
    headless: Optional[bool] = None,
    turbo_steps: Optional[int] = None,
    mute: Optional[bool] = None
) -> None:
    """
    Configure evaluation speed and I/O settings.
    
    Args:
        headless: If True, hide display window and skip all rendering
        turbo_steps: Number of physics steps per frame (higher = faster evaluation)
        mute: If True, disable all audio playback
    """
    global HEADLESS, TURBO_STEPS, MUTE, screen
    
    if headless is not None:
        HEADLESS = bool(headless)
    if turbo_steps is not None and turbo_steps >= 1:
        TURBO_STEPS = int(turbo_steps)
    if mute is not None:
        MUTE = bool(mute)
    
    # Recreate display for the new headless/visible mode
    try:
        flags = pygame.HIDDEN if HEADLESS else 0
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=flags)
    except Exception:
        pass


# ===================== GAME CONSTANTS =====================

# Screen dimensions
SCREEN_WIDTH: int = 400
SCREEN_HEIGHT: int = 600

# Physics constants
SPEED: int = 20              # Initial upward velocity when bird flaps
GRAVITY: float = 2.5         # Downward acceleration per frame
GAME_SPEED: int = 15         # Horizontal movement speed of pipes and ground

# Ground dimensions
GROUND_WIDTH: int = 2 * SCREEN_WIDTH
GROUND_HEIGHT: int = 100

# Pipe dimensions
PIPE_WIDTH: int = 80
PIPE_HEIGHT: int = 500
PIPE_GAP: int = 150          # Vertical gap between top and bottom pipes

# Audio file paths
WING_SOUND_PATH: str = 'assets/audio/wing.wav'
HIT_SOUND_PATH: str = 'assets/audio/hit.wav'

# ===================== AUDIO SYSTEM =====================

# Audio mixer initialization state
_mixer_ready: bool = False


def _ensure_mixer() -> None:
    """
    Initialize pygame mixer if not already done and audio is not muted.
    """
    global _mixer_ready
    if not _mixer_ready and not MUTE:
        try:
            pygame.mixer.init()
            _mixer_ready = True
        except Exception:
            _mixer_ready = False


def _play_sound(sound_path: str) -> None:
    """
    Play a sound effect if audio is enabled.
    
    Args:
        sound_path: Path to the sound file to play
    """
    if MUTE:
        return
    
    _ensure_mixer()
    if _mixer_ready:
        try:
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
        except Exception:
            pass

# ===================== GAME SPRITES =====================


class Bird(pygame.sprite.Sprite):
    """
    The player-controlled bird sprite.
    
    The bird has three animation states (upflap, midflap, downflap) and
    responds to gravity and flap commands. It maintains velocity and position
    for physics simulation and collision detection.
    """
    
    def __init__(self):
        """Initialize the bird sprite with animation frames and starting position."""
        pygame.sprite.Sprite.__init__(self)

        # Load bird animation frames
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
        ]

        # Physics properties
        self.speed: float = SPEED  # Current vertical velocity
        self.current_image: int = 0  # Current animation frame index
        
        # Set initial image and collision mask
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)

        # Set initial position (left side of screen, center vertically)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self) -> None:
        """
        Update bird physics and animation for each frame.
        
        Applies gravity, updates position, and cycles through animation frames.
        """
        # Cycle through animation frames
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        
        # Apply gravity and update position
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self) -> None:
        """
        Make the bird flap upward by setting negative velocity.
        """
        self.speed = -SPEED

    def begin(self) -> None:
        """
        Update animation during the splash screen phase.
        """
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

class Pipe(pygame.sprite.Sprite):
    """
    Obstacle pipe sprite that moves horizontally across the screen.
    
    Pipes come in pairs (top and bottom) with a gap between them.
    The bird must navigate through these gaps to score points.
    """
    
    def __init__(self, inverted: bool, xpos: int, ysize: int):
        """
        Initialize a pipe sprite.
        
        Args:
            inverted: If True, this is the top pipe (flipped vertically)
            xpos: Initial X position of the pipe
            ysize: Size of the pipe (height for bottom, determines gap for top)
        """
        pygame.sprite.Sprite.__init__(self)

        # Load and scale pipe image
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))

        # Set position
        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            # Top pipe: flip vertically and position above screen
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
            self.inverted = True
        else:
            # Bottom pipe: position from bottom of screen
            self.rect[1] = SCREEN_HEIGHT - ysize
            self.inverted = False

        # Create collision mask and scoring flag
        self.mask = pygame.mask.from_surface(self.image)
        self.counted = False  # Track if this pipe has been scored

    def update(self) -> None:
        """
        Move the pipe horizontally across the screen.
        """
        self.rect[0] -= GAME_SPEED

class Ground(pygame.sprite.Sprite):
    """
    Ground sprite that moves horizontally to create scrolling effect.
    
    Multiple ground sprites are used to create seamless scrolling.
    """
    
    def __init__(self, xpos: int):
        """
        Initialize ground sprite at specified position.
        
        Args:
            xpos: Initial X position of the ground sprite
        """
        pygame.sprite.Sprite.__init__(self)
        
        # Load and scale ground image
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        # Create collision mask
        self.mask = pygame.mask.from_surface(self.image)

        # Set position (ground is always at bottom of screen)
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self) -> None:
        """
        Move the ground horizontally across the screen.
        """
        self.rect[0] -= GAME_SPEED


# ===================== UTILITY FUNCTIONS =====================


def is_off_screen(sprite: pygame.sprite.Sprite) -> bool:
    """
    Check if a sprite has moved completely off the left side of the screen.
    
    Args:
        sprite: Sprite to check
        
    Returns:
        True if sprite is completely off-screen to the left
    """
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos: int) -> Tuple[Pipe, Pipe]:
    """
    Generate a random pair of pipes (top and bottom) at the specified position.
    
    Args:
        xpos: X position where the pipes should be created
        
    Returns:
        Tuple of (bottom_pipe, top_pipe)
    """
    # Random pipe size determines gap position
    size = random.randint(100, 300)
    
    # Create bottom pipe
    bottom_pipe = Pipe(False, xpos, size)
    
    # Create top pipe with matching gap
    top_pipe = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    
    return bottom_pipe, top_pipe

# ===================== OBSERVATION SYSTEM =====================


def _il_next_pipe_info(pipe_group: pygame.sprite.Group, bird_x: float) -> Tuple[float, float, float]:
    """
    Get information about the next upcoming pipe pair.
    
    Finds the first pipe pair that the bird hasn't passed yet and returns
    the pipe's X position and the gap boundaries.
    
    Args:
        pipe_group: Group containing all pipe sprites
        bird_x: Current X position of the bird
        
    Returns:
        Tuple of (pipe_x, gap_top_y, gap_bottom_y)
    """
    pipes = sorted(pipe_group.sprites(), key=lambda p: p.rect[0])
    
    for pipe in pipes:
        if hasattr(pipe, "inverted") and not pipe.inverted:
            # Check if this bottom pipe hasn't been passed yet
            if (pipe.rect[0] + PIPE_WIDTH) >= bird_x:
                # Find the matching top pipe at the same X position
                gap_top = 0
                for top_pipe in pipes:
                    if (getattr(top_pipe, "inverted", False) and 
                        abs(top_pipe.rect[0] - pipe.rect[0]) <= 1):
                        gap_top = top_pipe.rect[1] + top_pipe.rect[3]  # Bottom of top pipe
                        break
                
                gap_bottom = pipe.rect[1]  # Top of bottom pipe
                return float(pipe.rect[0]), float(gap_top), float(gap_bottom)
    
    # Fallback if no pipes ahead
    return float(SCREEN_WIDTH), 200.0, 200.0 + PIPE_GAP


def _il_make_obs(bird: Bird, pipe_group: pygame.sprite.Group) -> dict:
    """
    Create observation dictionary for imitation learning.
    
    Extracts relevant game state information that can be used by AI agents
    to make decisions about when to flap.
    
    Args:
        bird: The bird sprite
        pipe_group: Group containing all pipe sprites
        
    Returns:
        Dictionary containing observation features:
        - bird_y: Bird's vertical position
        - bird_v: Bird's vertical velocity
        - next_pipe_x: X position of next pipe
        - next_pipe_top: Top of the gap in next pipe pair
        - next_pipe_bottom: Bottom of the gap in next pipe pair
        - dist_to_pipe_center: Distance to center of next pipe
    """
    bird_y = float(bird.rect[1])
    bird_v = float(bird.speed)
    bird_x = float(bird.rect[0])
    
    # Get next pipe information
    pipe_x, gap_top, gap_bottom = _il_next_pipe_info(pipe_group, bird_x)
    
    # Calculate distance to pipe center
    bird_center_x = bird_x + bird.rect[2] * 0.5
    pipe_center_x = pipe_x + PIPE_WIDTH * 0.5
    dist_to_center = float(pipe_center_x - bird_center_x)
    
    return {
        "bird_y": bird_y,
        "bird_v": bird_v,
        "next_pipe_x": pipe_x,
        "next_pipe_top": gap_top,
        "next_pipe_bottom": gap_bottom,
        "dist_to_pipe_center": dist_to_center,
    }

# ===================== GAME INITIALIZATION =====================

# Initialize pygame
pygame.init()

# Create display surface
flags = pygame.HIDDEN if HEADLESS else 0
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=flags)
pygame.display.set_caption('Flappy Bird')

# Load and scale background images
BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

# Game clock for frame rate control
clock = pygame.time.Clock()

# ===================== GAME WORLD MANAGEMENT =====================


def _new_world() -> Tuple[Bird, pygame.sprite.Group, pygame.sprite.Group, pygame.sprite.Group]:
    """
    Create a fresh game world with all sprites for a new episode.
    
    Returns:
        Tuple of (bird, bird_group, ground_group, pipe_group)
    """
    # Create bird
    bird_group = pygame.sprite.Group()
    bird = Bird()
    bird_group.add(bird)

    # Create ground sprites for seamless scrolling
    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground = Ground(GROUND_WIDTH * i)
        ground_group.add(ground)

    # Create initial pipe pairs
    pipe_group = pygame.sprite.Group()
    for i in range(2):
        pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    return bird, bird_group, ground_group, pipe_group


def _render_splash(bird_group: pygame.sprite.Group, ground_group: pygame.sprite.Group) -> None:
    """
    Render the splash screen with start message.
    
    Args:
        bird_group: Group containing bird sprites
        ground_group: Group containing ground sprites
    """
    if HEADLESS:
        return
    
    screen.blit(BACKGROUND, (0, 0))
    screen.blit(BEGIN_IMAGE, (120, 150))
    bird_group.draw(screen)
    ground_group.draw(screen)
    pygame.display.update()


def _render_game(
    bird_group: pygame.sprite.Group,
    pipe_group: pygame.sprite.Group,
    ground_group: pygame.sprite.Group
) -> None:
    """
    Render the main game screen.
    
    Args:
        bird_group: Group containing bird sprites
        pipe_group: Group containing pipe sprites
        ground_group: Group containing ground sprites
    """
    if HEADLESS:
        return
    
    screen.blit(BACKGROUND, (0, 0))
    bird_group.draw(screen)
    pipe_group.draw(screen)
    ground_group.draw(screen)
    pygame.display.update()


def _tick() -> None:
    """
    Control frame rate based on evaluation mode.
    """
    # Keep pygame responsive even in fast mode
    clock.tick(300 if TURBO_STEPS > 1 or HEADLESS else 15)


def _update_world(
    bird_group: pygame.sprite.Group,
    ground_group: pygame.sprite.Group,
    pipe_group: pygame.sprite.Group,
    bird_x_fixed: int
) -> None:
    """
    Update the game world by managing off-screen sprites.
    
    Args:
        bird_group: Group containing bird sprites
        ground_group: Group containing ground sprites
        pipe_group: Group containing pipe sprites
        bird_x_fixed: Fixed X position of bird for reference
    """
    # Replace off-screen ground sprites
    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])
        ground_group.add(Ground(GROUND_WIDTH - 20))

    # Replace off-screen pipe pairs
    if is_off_screen(pipe_group.sprites()[0]):
        # Remove both pipes in the pair
        pipe_group.remove(pipe_group.sprites()[0])
        pipe_group.remove(pipe_group.sprites()[0])
        
        # Add new pipe pair
        pipes = get_random_pipes(SCREEN_WIDTH * 2)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])


def _update_score(pipe_group: pygame.sprite.Group, bird_x_fixed: int, current_score: int) -> int:
    """
    Update the score when pipes are passed.
    
    Args:
        pipe_group: Group containing pipe sprites
        bird_x_fixed: Fixed X position of bird for scoring reference
        current_score: Current score
        
    Returns:
        Updated score
    """
    # Count score when the lower pipe passes the bird's fixed X position
    for pipe in pipe_group.sprites():
        if (hasattr(pipe, "inverted") and not pipe.inverted and not pipe.counted):
            if (pipe.rect[0] + PIPE_WIDTH) < bird_x_fixed:
                pipe.counted = True
                current_score += 1
    
    return current_score


def _check_collisions(
    bird_group: pygame.sprite.Group,
    ground_group: pygame.sprite.Group,
    pipe_group: pygame.sprite.Group
) -> bool:
    """
    Check for collisions between bird and obstacles.
    
    Args:
        bird_group: Group containing bird sprites
        ground_group: Group containing ground sprites
        pipe_group: Group containing pipe sprites
        
    Returns:
        True if collision detected, False otherwise
    """
    # Check collision with ground
    ground_collision = pygame.sprite.groupcollide(
        bird_group, ground_group, False, False, pygame.sprite.collide_mask
    )
    
    # Check collision with pipes
    pipe_collision = pygame.sprite.groupcollide(
        bird_group, pipe_group, False, False, pygame.sprite.collide_mask
    )
    
    return bool(ground_collision or pipe_collision)

# ===================== MAIN GAME LOOP =====================


def run_one_episode(
    headless: Optional[bool] = None,
    turbo_steps: Optional[int] = None,
    mute: Optional[bool] = None
) -> int:
    """
    Run one complete episode of Flappy Bird and return the final score.
    
    An episode consists of:
    1. Splash screen phase (waiting for input to start)
    2. Main game phase (flying through pipes until collision)
    3. Death phase (brief pause before returning)
    
    Args:
        headless: If True, hide window and skip rendering
        turbo_steps: Number of physics steps per frame (higher = faster)
        mute: If True, disable audio
        
    Returns:
        Final score (number of pipes passed)
    """
    # Apply evaluation options if provided
    if headless is not None or turbo_steps is not None or mute is not None:
        set_eval_options(headless=headless, turbo_steps=turbo_steps, mute=mute)

    # Safety: Force normal speed for human play
    global TURBO_STEPS
    if not HEADLESS and TURBO_STEPS != 1:
        TURBO_STEPS = 1
        try:
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0)
            pygame.display.set_caption('Flappy Bird')
        except Exception:
            pass

    global LAST_SCORE
    
    # Initialize game world
    bird, bird_group, ground_group, pipe_group = _new_world()
    score = 0
    bird_x_fixed = int(bird.rect[0])  # Fixed reference point for scoring

    # Notify external controller of episode start
    if EXTERNAL_CONTROLLER and hasattr(EXTERNAL_CONTROLLER, "start_episode"):
        EXTERNAL_CONTROLLER.start_episode()

    # ===== SPLASH SCREEN PHASE =====
    begin = True
    begin_frames = 0

    while begin:
        _tick()

        # Process keyboard input
        human_flap = False
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == K_SPACE or event.key == K_UP:
                    human_flap = True

        # Also poll current key state (covers missed KEYDOWN events)
        if not HEADLESS:
            keys = pygame.key.get_pressed()
            human_flap = human_flap or keys[K_SPACE] or keys[K_UP]

        # Determine action (default to human input)
        action = 1 if human_flap else 0

        # Build observation and allow controller to override
        obs = _il_make_obs(bird, pipe_group)
        if EXTERNAL_CONTROLLER and hasattr(EXTERNAL_CONTROLLER, "decide"):
            override = EXTERNAL_CONTROLLER.decide(obs)
            if override in (0, 1):
                action = override

        # Auto-start for AI agents if they haven't acted yet
        begin_frames += 1
        if (EXTERNAL_CONTROLLER is not None and action == 0 and 
            begin_frames >= AUTO_START_DELAY_FRAMES):
            if not hasattr(EXTERNAL_CONTROLLER, "auto_start") or EXTERNAL_CONTROLLER.auto_start:
                action = 1

        # Execute flap action
        if action == 1:
            bird.bump()
            _play_sound(WING_SOUND_PATH)
            begin = False  # Start main game

        # Record observation-action pair
        if EXTERNAL_CONTROLLER and hasattr(EXTERNAL_CONTROLLER, "record"):
            EXTERNAL_CONTROLLER.record(obs, int(action))

        # Update world during splash screen
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_WIDTH - 20))
        
        bird.begin()
        ground_group.update()
        _render_splash(bird_group, ground_group)

    # ===== MAIN GAME PHASE =====
    alive = True
    while alive:
        _tick()

        # Process keyboard input
        human_flap = False
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                LAST_SCORE = score
                return score
            if event.type == pygame.KEYDOWN:
                if event.key == K_SPACE or event.key == K_UP:
                    human_flap = True

        if not HEADLESS:
            keys = pygame.key.get_pressed()
            human_flap = human_flap or keys[K_SPACE] or keys[K_UP]

        # Run multiple physics steps per frame for turbo mode
        for _ in range(TURBO_STEPS):
            # Determine action (default to human input)
            action = 1 if human_flap else 0

            # Build observation and allow controller to override
            obs = _il_make_obs(bird, pipe_group)
            if EXTERNAL_CONTROLLER and hasattr(EXTERNAL_CONTROLLER, "decide"):
                override = EXTERNAL_CONTROLLER.decide(obs)
                if override in (0, 1):
                    action = override

            # Execute flap action
            if action == 1:
                bird.bump()
                if TURBO_STEPS == 1:  # Avoid audio spam in turbo mode
                    _play_sound(WING_SOUND_PATH)

            # Update game world
            _update_world(bird_group, ground_group, pipe_group, bird_x_fixed)

            # Update score
            score = _update_score(pipe_group, bird_x_fixed, score)

            # Update all sprites
            bird_group.update()
            ground_group.update()
            pipe_group.update()

            # Record observation-action pair
            if EXTERNAL_CONTROLLER and hasattr(EXTERNAL_CONTROLLER, "record"):
                EXTERNAL_CONTROLLER.record(obs, int(action))

            # Check for collisions
            if _check_collisions(bird_group, ground_group, pipe_group):
                _play_sound(HIT_SOUND_PATH)
                if not HEADLESS:
                    time.sleep(1)
                alive = False
                break

        _render_game(bird_group, pipe_group, ground_group)

    # Episode finished
    LAST_SCORE = score
    
    # Notify external controller of episode end
    if EXTERNAL_CONTROLLER and hasattr(EXTERNAL_CONTROLLER, "end_episode"):
        EXTERNAL_CONTROLLER.end_episode(score)
    
    return score


# ===================== MAIN ENTRY POINT =====================


def main() -> None:
    """
    Run a single episode with human-visible settings and print the score.
    
    This is the main entry point when running the script directly.
    """
    # Ensure human-friendly settings
    set_eval_options(headless=False, turbo_steps=1, mute=False)
    
    # Run one episode
    score = run_one_episode()
    print(f"Episode score: {score}")


if __name__ == '__main__':
    main()
