#Standard library imports


#Third-party imports (np -> model -> env -> visualization)
import numpy as np
import gymnasium as gym

from typing import Tuple, List, Optional

#Suppress gym warnings
gym.logger.set_level(40)


#seed settings
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_env_seed(env: gym.Env, seed: int = SEED) -> None:
    """Set the seed for the environment for reproducibility."""
    env.reset(seed=seed)
    env.action_space.seed(seed)

