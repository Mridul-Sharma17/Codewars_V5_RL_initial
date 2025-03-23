"""
Tower Defense RL - Battle Simulator

This module handles running battles between different strategies.
"""

import os
import sys
import time
import logging
import copy
import random
import uuid
import importlib.util
import shutil
from typing import Dict, List, Tuple, Any

from rl_config import *

logger = logging.getLogger('battle_simulator')

class BattleSimulator:
    """Runs battles between tower defense strategies."""
    
    def __init__(self):
        """Initialize the battle simulator."""
        self.game_config_path = os.path.join(GAME_ENGINE_DIR, "config.py")
        self.game_main_path = os.path.join(GAME_ENGINE_DIR, "main.py")
        self.backup_config_path = os.path.join(TEMP_DIR, "config.py.bak")
        self.teams_helper_path = os.path.join(GAME_ENGINE_DIR, "teams")
        self.temp_dir = TEMP_DIR
        
        # Create a proper default config if needed
        if not os.path.exists(self.backup_config_path):
            self._create_default_config_backup()
            
    def _create_default_config_backup(self):
        """Create a proper default config backup with correct imports."""
        default_config = """import sys
import os

# Add path to teams directory and baselines directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "teams", "baselines"))

# Import pratyaksh strategy
from pratyaksh import *

# Set teams to pratyaksh by default
TEAM1 = pratyaksh
TEAM2 = pratyaksh
VALUE_ERROR = False
"""
        # Write the default config to the backup file
        with open(self.backup_config_path, 'w') as f:
            f.write(default_config)
    
    def run_tournament(self, strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Run a round-robin tournament between all strategies.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            
        Returns:
            Dictionary of battle_id -> battle_data
        """
        # Prepare a list of strategies that will participate in the tournament
        participants = list(strategies.values())
        logger.info(f"Running tournament with {len(participants)} strategies")
        
        # Make sure all strategies have metrics
        for strategy in participants:
            if "metrics" not in strategy:
                strategy["metrics"] = {
                    "wins": 0,
                    "losses": 0,
                    "games_played": 0,
                    "win_rate": 0.0,
                    "avg_tower_health": 0.0,
                    "fitness": 0.0
                }
        
        # Prepare results dictionary
        battle_results = {}
        
        # Run all possible pairings
        battle_counter = 0
        total_battles = len(participants) * (len(participants) - 1)
        
        for i, strategy1 in enumerate(participants):
            for j, strategy2 in enumerate(participants):
                if i == j:  # Skip battles against self
                    continue
                
                battle_counter += 1
                logger.info(f"Battle {battle_counter}/{total_battles}: {strategy1['name']} vs {strategy2['name']}")
                
                # Run the battle
                battle_id = f"battle_{uuid.uuid4().hex}"
                result = self._run_battle(strategy1, strategy2)
                
                # Store the result
                battle_results[battle_id] = {
                    "battle_id": battle_id,
                    "strategy1_id": strategy1["id"],
                    "strategy2_id": strategy2["id"],
                    "winner_id": result["winner_id"],
                    "loser_id": result["loser_id"],
                    "strategy1_health": result["strategy1_health"],
                    "strategy2_health": result["strategy2_health"],
                    "timestamp": result["timestamp"]
                }
                
                # Update strategy metrics
                if strategy1["id"] == result["winner_id"]:
                    strategy1["metrics"]["wins"] += 1
                    strategy2["metrics"]["losses"] += 1
                elif strategy2["id"] == result["winner_id"]:
                    strategy2["metrics"]["wins"] += 1
                    strategy1["metrics"]["losses"] += 1
                
                strategy1["metrics"]["games_played"] += 1
                strategy2["metrics"]["games_played"] += 1
                
                # Update average tower health
                if result["strategy1_health"] is not None:
                    current_avg = strategy1["metrics"]["avg_tower_health"]
                    games_played = strategy1["metrics"]["games_played"]
                    strategy1["metrics"]["avg_tower_health"] = (
                        (current_avg * (games_played - 1) + result["strategy1_health"]) / games_played
                    )
                
                if result["strategy2_health"] is not None:
                    current_avg = strategy2["metrics"]["avg_tower_health"]
                    games_played = strategy2["metrics"]["games_played"]
                    strategy2["metrics"]["avg_tower_health"] = (
                        (current_avg * (games_played - 1) + result["strategy2_health"]) / games_played
                    )
                
                # Update win rate
                for s in [strategy1, strategy2]:
                    wins = s["metrics"]["wins"]
                    games = s["metrics"]["games_played"]
                    s["metrics"]["win_rate"] = wins / games if games > 0 else 0
        
        logger.info(f"Tournament completed. {len(battle_results)} battles run.")
        return battle_results
    
    def _run_battle(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a battle between two strategies.
        
        Args:
            strategy1: First strategy data
            strategy2: Second strategy data
            
        Returns:
            Battle result data
        """
        # Create temporary copies of strategies if needed
        strategy1_path = self._prepare_strategy_file(strategy1)
        strategy2_path = self._prepare_strategy_file(strategy2)
        
        # Modify game config to use these strategies
        self._update_game_config(strategy1_path, strategy2_path)
        
        # Run the game
        winner, strategy1_health, strategy2_health = self._run_game()
        
        # Determine winner and loser IDs
        if winner == 1:
            winner_id = strategy1["id"]
            loser_id = strategy2["id"]
        elif winner == 2:
            winner_id = strategy2["id"]
            loser_id = strategy1["id"]
        else:  # Tie
            winner_id = None
            loser_id = None
        
        # Clean up
        self._restore_game_config()
        
        return {
            "winner_id": winner_id,
            "loser_id": loser_id,
            "strategy1_health": strategy1_health,
            "strategy2_health": strategy2_health,
            "timestamp": time.time()
        }
    
    def _prepare_strategy_file(self, strategy: Dict[str, Any]) -> str:
        """
        Prepare a strategy file for battle.
        
        If the strategy has a source_file, just return that path.
        Otherwise, generate a temporary file.
        
        Args:
            strategy: Strategy data
            
        Returns:
            Path to the strategy file
        """
        if "source_file" in strategy and os.path.exists(strategy["source_file"]):
            return strategy["source_file"]
        
        # Should never happen, but just in case
        logger.warning(f"Strategy {strategy['id']} has no source file. Generating temporary file.")
        return self._generate_temporary_strategy_file(strategy)
    
    def _generate_temporary_strategy_file(self, strategy: Dict[str, Any]) -> str:
        """Generate a temporary strategy file."""
        # This should only happen if something went wrong with strategy generation
        from adaptive_templates import AdaptiveTemplates
        templates = AdaptiveTemplates()
        
        # Generate code
        code = templates.generate_strategy_code(
            strategy["template_type"], 
            strategy["parameters"],
            strategy["name"]
        )
        
        # Write to temporary file
        temp_file = os.path.join(TEMP_DIR, f"{strategy['id']}.py")
        with open(temp_file, 'w') as f:
            f.write(code)
        
        return temp_file
    
    def _update_game_config(self, strategy1_path: str, strategy2_path: str) -> None:
        """
        Update game config to use specific strategies.
        
        Args:
            strategy1_path: Path to first strategy file
            strategy2_path: Path to second strategy file
        """
        # Get module names from file paths
        strategy1_module = os.path.splitext(os.path.basename(strategy1_path))[0]
        strategy2_module = os.path.splitext(os.path.basename(strategy2_path))[0]
        
        # Get directory containing the strategy files
        strategy1_dir = os.path.dirname(strategy1_path)
        strategy2_dir = os.path.dirname(strategy2_path)
        
        # Create new config content with precise import paths
        config_content = f"""import sys
import os

# Add paths to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # game_engine dir
sys.path.append(r"{self.teams_helper_path}")  # teams dir
sys.path.append(r"{strategy1_dir}")  # Strategy 1 dir
sys.path.append(r"{strategy2_dir}")  # Strategy 2 dir

# Import strategy modules
try:
    from {strategy1_module} import *
except ImportError as e:
    print(f"Error importing {strategy1_module}: {{e}}")
    raise

try:
    from {strategy2_module} import *
except ImportError as e:
    print(f"Error importing {strategy2_module}: {{e}}")
    raise

# Define teams
TEAM1 = {strategy1_module}
TEAM2 = {strategy2_module}
VALUE_ERROR = False
"""
        
        # Write to config file
        with open(self.game_config_path, 'w') as f:
            f.write(config_content)
    
    def _restore_game_config(self) -> None:
        """Restore original game config."""
        shutil.copy(self.backup_config_path, self.game_config_path)
    
    def _run_game(self) -> Tuple[int, float, float]:
        """
        Run the game and return the winner.
        
        Returns:
            Tuple of (winner, strategy1_health, strategy2_health)
            winner: 1 for strategy1, 2 for strategy2, 0 for tie
        """
        # Save current working directory and path
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            # Change working directory to the game engine directory
            # This is critical for relative path resolution
            os.chdir(GAME_ENGINE_DIR)
            
            # Set up Python path for correct imports
            sys.path = [
                GAME_ENGINE_DIR,  # First, to make internal imports work
                os.path.dirname(GAME_ENGINE_DIR),  # Parent directory
                self.teams_helper_path,  # Teams directory for helper_function
            ] + [p for p in sys.path if p not in [GAME_ENGINE_DIR, os.path.dirname(GAME_ENGINE_DIR)]]
            
            # Import the game modules properly
            spec = importlib.util.spec_from_file_location("game_main", self.game_main_path)
            game_main = importlib.util.module_from_spec(spec)
            
            try:
                spec.loader.exec_module(game_main)
            except Exception as e:
                logger.error(f"Error loading game_main: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return 0, None, None  # Tie due to error
            
            # Define constants for the game
            GAME_START_TIME = 0
            GAME_END_TIME = 3600
            
            # Modified version of the game's main function that returns battle result
            def modified_main():
                try:
                    # Import the required modules from the game
                    from game import Game
                    
                    # Get the teams from config
                    from config import TEAM1, TEAM2
                    
                    # Validate teams
                    team1_valid = game_main.validate_module(TEAM1, "TEAM 1")
                    team2_valid = game_main.validate_module(TEAM2, "TEAM 2")
                    
                    if not (team1_valid and team2_valid):
                        logger.error("Team validation failed")
                        return 0, 0, 0  # Tie due to validation failure
                    
                    # Create game instance
                    game = Game(TEAM1.troops, TEAM2.troops, TEAM1.team_name, TEAM2.team_name)
                    
                    # Run the game silently (no UI)
                    winner, team1_health, team2_health = run_headless_game(game)
                    
                    return winner, team1_health, team2_health
                except Exception as e:
                    logger.error(f"Error in modified_main: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return 0, 0, 0  # Tie due to error
            
            def run_headless_game(game):
                """Run the game without UI and return the winner."""
                try:
                    # Import necessary modules
                    from scripts.dataflow import DataFlow
                    
                    # Initialize variables
                    game_counter = 0
                    tower1_health = game.tower1.health
                    tower2_health = game.tower2.health
                    
                    # Run game loop
                    while game_counter < GAME_END_TIME and tower1_health > 0 and tower2_health > 0:
                        # Update game state
                        if GAME_END_TIME > game_counter >= GAME_START_TIME:
                            # Provide data to teams
                            game.data_provided1 = {}
                            game.data_provided2 = {}
                            
                            # Create dummy objects for troops and towers
                            DataFlow.provide_data(game)
                            
                            # Get deployment decisions
                            DataFlow.deployment(game)
                            
                            # Update game state
                            DataFlow.attack_die(game)
                        
                        # Update counter and health
                        game_counter += 1
                        tower1_health = game.tower1.health
                        tower2_health = game.tower2.health
                    
                    # Determine winner
                    if tower1_health <= 0 and tower2_health <= 0:
                        return 0, 0, 0  # Tie
                    elif tower1_health <= 0:
                        return 2, 0, tower2_health  # Team 2 wins
                    elif tower2_health <= 0:
                        return 1, tower1_health, 0  # Team 1 wins
                    elif tower1_health > tower2_health:
                        return 1, tower1_health, tower2_health  # Team 1 wins (tiebreaker)
                    elif tower2_health > tower1_health:
                        return 2, tower1_health, tower2_health  # Team 2 wins (tiebreaker)
                    else:
                        return 0, tower1_health, tower2_health  # Tie
                except Exception as e:
                    logger.error(f"Error in run_headless_game: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return 0, 0, 0  # Tie due to error
            
            # Run the modified main function
            try:
                winner, strategy1_health, strategy2_health = modified_main()
                return winner, strategy1_health, strategy2_health
            except Exception as e:
                logger.error(f"Error running game: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return 0, None, None  # Tie due to error
        
        finally:
            # Restore original working directory and path
            os.chdir(original_cwd)
            sys.path = original_path