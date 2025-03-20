"""
Battle simulator for tower defense reinforcement learning system.
Simulates battles between strategies to determine their effectiveness.
"""

import os
import sys
import json
import random
import subprocess
import tempfile
import time
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_data/battle_simulator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BattleSimulator")

class BattleSimulator:
    """
    Simulates battles between tower defense strategies to evaluate their performance.
    """
    
    def __init__(self, game_executable="./battle_simulator"):
        """
        Initialize the battle simulator.
        
        Args:
            game_executable: Path to the game executable
        """
        self.game_executable = game_executable
        self.strategy_dir = "teams"
        self.temp_dir = "temp_strategies"
        self.visualize = False
        self.battle_timeout = 300  # Maximum seconds per battle
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs("training_data", exist_ok=True)
    
    def run_battles(self, 
                   strategies: Dict[str, Any], 
                   strategy_generator,
                   num_battles: int, 
                   tournament_style: bool = False) -> List[Dict[str, Any]]:
        """
        Run a series of battles between strategies.
        
        Args:
            strategies: Dictionary of strategies to battle
            strategy_generator: Generator to produce strategy code
            num_battles: Number of battles to run
            tournament_style: Whether to use tournament-style matchups
            
        Returns:
            List of battle results
        """
        # Generate all needed strategy files
        self._generate_strategy_files(strategies, strategy_generator)
        
        # Generate battle pairings
        battle_pairs = self._generate_battle_pairs(strategies, num_battles, tournament_style)
        
        # Run battles
        battle_results = []
        
        for i, (strategy_a_id, strategy_b_id) in enumerate(battle_pairs):
            logger.info(f"Running battle {i+1}/{len(battle_pairs)}: {strategy_a_id} vs {strategy_b_id}")
            
            result = self._run_battle(strategy_a_id, strategy_b_id)
            
            if result:
                battle_results.append(result)
                logger.info(f"Battle result: {result['winner']} defeated {result['loser']} "
                           f"with {result['winner_tower_health']} health remaining")
            else:
                logger.warning(f"Battle failed: {strategy_a_id} vs {strategy_b_id}")
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        return battle_results
    
    def _generate_strategy_files(self, strategies: Dict[str, Any], strategy_generator):
        """
        Generate Python files for all strategies.
        
        Args:
            strategies: Dictionary of strategies to generate files for
            strategy_generator: Generator to produce strategy code
        """
        for strategy_id, strategy in strategies.items():
            try:
                # Generate strategy code
                strategy_code = strategy_generator.generate_strategy_code(strategy)
                
                # Save to file
                file_path = os.path.join(self.temp_dir, f"{strategy_id}.py")
                with open(file_path, "w") as f:
                    f.write(strategy_code)
                    
                logger.debug(f"Generated strategy file for {strategy_id}")
            except Exception as e:
                logger.error(f"Error generating strategy file for {strategy_id}: {e}")
    
    def _generate_battle_pairs(self, 
                              strategies: Dict[str, Any], 
                              num_battles: int,
                              tournament_style: bool) -> List[Tuple[str, str]]:
        """
        Generate pairs of strategies to battle.
        
        Args:
            strategies: Dictionary of strategies
            num_battles: Number of battles to generate
            tournament_style: Whether to use tournament-style matchups
            
        Returns:
            List of (strategy_a_id, strategy_b_id) tuples
        """
        strategy_ids = list(strategies.keys())
        
        if len(strategy_ids) < 2:
            logger.error("Not enough strategies to battle")
            return []
        
        battle_pairs = []
        
        if tournament_style:
            # Tournament style - each strategy battles each other strategy
            for i in range(len(strategy_ids)):
                for j in range(i + 1, len(strategy_ids)):
                    battle_pairs.append((strategy_ids[i], strategy_ids[j]))
                    
            # If we need more battles than the full tournament, repeat some matchups
            if len(battle_pairs) < num_battles:
                additional_pairs = []
                while len(additional_pairs) + len(battle_pairs) < num_battles:
                    # Random matchups for additional battles
                    i, j = random.sample(range(len(strategy_ids)), 2)
                    additional_pairs.append((strategy_ids[i], strategy_ids[j]))
                
                battle_pairs.extend(additional_pairs)
        else:
            # Random matchups
            while len(battle_pairs) < num_battles:
                # Try to ensure each strategy gets battled at least once
                unused_strategies = [s for s in strategy_ids if not any(s in pair for pair in battle_pairs)]
                
                if len(unused_strategies) >= 2:
                    # Pair two unused strategies
                    i, j = random.sample(range(len(unused_strategies)), 2)
                    battle_pairs.append((unused_strategies[i], unused_strategies[j]))
                elif len(unused_strategies) == 1:
                    # Pair the remaining unused strategy with a random one
                    unused = unused_strategies[0]
                    other = random.choice([s for s in strategy_ids if s != unused])
                    battle_pairs.append((unused, other))
                else:
                    # All strategies used, create random pairs
                    i, j = random.sample(range(len(strategy_ids)), 2)
                    battle_pairs.append((strategy_ids[i], strategy_ids[j]))
        
        # Trim to requested number of battles
        return battle_pairs[:num_battles]
    
    def _run_battle(self, strategy_a_id: str, strategy_b_id: str) -> Optional[Dict[str, Any]]:
        """
        Run a battle between two strategies.
        
        Args:
            strategy_a_id: ID of first strategy
            strategy_b_id: ID of second strategy
            
        Returns:
            Battle result dictionary or None if battle failed
        """
        # Prepare paths to strategy files
        strategy_a_path = os.path.join(self.temp_dir, f"{strategy_a_id}.py")
        strategy_b_path = os.path.join(self.temp_dir, f"{strategy_b_id}.py")
        
        # Check if strategy files exist
        if not os.path.exists(strategy_a_path) or not os.path.exists(strategy_b_path):
            logger.error(f"Strategy files not found: {strategy_a_path} or {strategy_b_path}")
            return None
        
        # Create temporary results file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            results_path = temp_file.name
        
        try:
            # Build command to run battle
            cmd = [
                self.game_executable,
                "--team1", strategy_a_path,
                "--team2", strategy_b_path,
                "--results", results_path,
                "--headless", "true" if not self.visualize else "false",
                "--fast-forward", "true"
            ]
            
            # Run the battle process with timeout
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            try:
                stdout, stderr = process.communicate(timeout=self.battle_timeout)
                
                # Check for errors
                if process.returncode != 0:
                    logger.error(f"Battle process failed with code {process.returncode}")
                    logger.error(f"Stderr: {stderr.decode('utf-8', errors='ignore')}")
                    return None
                
            except subprocess.TimeoutExpired:
                logger.error(f"Battle timed out after {self.battle_timeout} seconds")
                process.kill()
                return None
            
            # Parse battle results
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Determine winner and loser
                winner_id = None
                loser_id = None
                winner_tower_health = 0
                loser_tower_health = 0
                
                if results["team1_won"]:
                    winner_id = strategy_a_id
                    loser_id = strategy_b_id
                    winner_tower_health = results["team1_tower_health"]
                    loser_tower_health = results["team2_tower_health"]
                else:
                    winner_id = strategy_b_id
                    loser_id = strategy_a_id
                    winner_tower_health = results["team2_tower_health"]
                    loser_tower_health = results["team1_tower_health"]
                
                # Create battle result object
                battle_result = {
                    "winner": winner_id,
                    "loser": loser_id,
                    "winner_tower_health": winner_tower_health,
                    "loser_tower_health": loser_tower_health,
                    "game_duration": results.get("game_duration", 0),
                    "timestamp": time.time()
                }
                
                return battle_result
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error parsing battle results: {e}")
                return None
            
        finally:
            # Clean up temporary results file
            if os.path.exists(results_path):
                try:
                    os.unlink(results_path)
                except:
                    pass
    
    def _cleanup_temp_files(self):
        """Clean up temporary strategy files."""
        try:
            for filename in os.listdir(self.temp_dir):
                if filename.endswith('.py'):
                    filepath = os.path.join(self.temp_dir, filename)
                    os.unlink(filepath)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")
    
    def enable_visualization(self, enable: bool = True):
        """Enable or disable battle visualization."""
        self.visualize = enable
    
    def set_timeout(self, timeout_seconds: int):
        """Set battle timeout in seconds."""
        self.battle_timeout = max(10, timeout_seconds)