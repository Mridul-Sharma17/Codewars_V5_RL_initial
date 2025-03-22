"""
Tower Defense RL - Configuration Settings
"""

import os

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAME_DIR = os.path.join(BASE_DIR, "game")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
EVOLVED_STRATEGIES_DIR = os.path.join(BASE_DIR, "evolved_strategies")
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
CHECKPOINT_DIR = os.path.join(TRAINING_DATA_DIR, "checkpoints")
REPORT_DIR = os.path.join(TRAINING_DATA_DIR, "reports")
VISUALIZATION_DIR = os.path.join(TRAINING_DATA_DIR, "visualizations")
LOG_DIR = os.path.join(TRAINING_DATA_DIR, "logs")
TEMP_DIR = os.path.join(TRAINING_DATA_DIR, "temp")

# Create directories if they don't exist
for directory in [
    EVOLVED_STRATEGIES_DIR, TRAINING_DATA_DIR, CHECKPOINT_DIR, 
    REPORT_DIR, VISUALIZATION_DIR, LOG_DIR, TEMP_DIR
]:
    os.makedirs(directory, exist_ok=True)

# Evolution Settings
POPULATION_SIZE = 20
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.7
INITIAL_MUTATION_RATE = 0.3
REFINEMENT_MUTATION_RATE = 0.2
SPECIALIZATION_MUTATION_RATE = 0.1

# Tournament Settings
EXPLORATION_PHASE_END = 6  # Generations
REFINEMENT_PHASE_END = 11  # Generations
MAX_GENERATIONS = 20
BATTLE_TIMEOUT = 300  # seconds

# Strategy Template Weights
# Chances of each template type being used in initial population
TEMPLATE_WEIGHTS = {
    "counter_picker": 0.3,
    "resource_manager": 0.3,
    "lane_adaptor": 0.2,
    "phase_based": 0.2
}

# Fitness Function Weights
FITNESS_WEIGHTS = {
    "wilson_score": 0.4,
    "elite_performance": 0.3,
    "tower_health": 0.2,
    "versatility": 0.1
}

# Checkpoint Settings
CHECKPOINTS_TO_KEEP = 3
CHECKPOINT_FREQUENCY = 1  # Save checkpoint every N generations

# Baseline Strategies
BASELINE_STRATEGIES = [
    {
        "id": "pratyaksh",
        "name": "Pratyaksh",
        "source_file": os.path.join(GAME_DIR, "teams", "baselines", "pratyaksh.py"),
        "template_type": "direct_code",
        "is_static": True
    }
]

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL