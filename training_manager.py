"""
Training manager for tower defense reinforcement learning system.
Manages the training process, including strategy generation, evolution, and persistence.
"""

import os
import json
import time
import random
import shutil
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from strategy_generator import StrategyGenerator
from battle_simulator import BattleSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_data/training_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainingManager")

class TrainingManager:
    """
    Manages the reinforcement learning training process for tower defense strategies.
    Coordinates strategy generation, battles, evolution, and checkpoint saving.
    """
    
    def __init__(self, 
                config_path: str = "training_config.json", 
                resume_from_checkpoint: bool = True):
        """
        Initialize the training manager.
        
        Args:
            config_path: Path to training configuration file
            resume_from_checkpoint: Whether to resume from the latest checkpoint
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up directories
        self.data_dir = self.config.get("data_directory", "training_data")
        self.checkpoint_dir = os.path.join(self.data_dir, "checkpoints")
        self.results_dir = os.path.join(self.data_dir, "results")
        self.best_strategies_dir = os.path.join(self.data_dir, "best_strategies")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.best_strategies_dir, exist_ok=True)
        
        # Create components
        self.strategy_generator = StrategyGenerator(
            template_type=self.config.get("default_template", "adaptive"),
            mutation_rate=self.config.get("mutation_rate", 0.25)
        )
        
        self.battle_simulator = BattleSimulator(
            game_executable=self.config.get("game_executable", "./battle_simulator")
        )
        
        # Training state
        self.current_generation = 0
        self.strategies = {}
        self.battle_history = []
        self.metrics_history = {
            "avg_win_rate": [],
            "best_win_rate": [],
            "diversity": [],
            "generation_times": []
        }
        
        # Resume from checkpoint if enabled
        if resume_from_checkpoint:
            self._load_latest_checkpoint()
    
    def train(self, generations: int = None) -> Dict[str, Any]:
        """
        Run the training process for a specified number of generations.
        
        Args:
            generations: Number of generations to train (if None, use config value)
            
        Returns:
            Dictionary containing the best strategies found
        """
        if generations is None:
            generations = self.config.get("generations", 10)
        
        starting_generation = self.current_generation
        population_size = self.config.get("population_size", 30)
        battles_per_generation = self.config.get("battles_per_generation", 50)
        tournament_style = self.config.get("tournament_style", False)
        checkpoint_interval = self.config.get("checkpoint_interval", 1)
        visualize_battles = self.config.get("visualize_battles", False)
        
        # Set battle visualization
        self.battle_simulator.enable_visualization(visualize_battles)
        
        logger.info(f"Starting training from generation {starting_generation} for {generations} generations")
        logger.info(f"Population size: {population_size}, Battles per generation: {battles_per_generation}")
        
        # If this is the first generation and we don't have strategies yet, generate them
        if self.current_generation == 0 and not self.strategies:
            logger.info("Generating initial population")
            self.strategies = self.strategy_generator.generate_initial_population(population_size)
        
        try:
            # Main training loop
            for gen in range(starting_generation, starting_generation + generations):
                self.current_generation = gen
                generation_start_time = time.time()
                
                logger.info(f"===== Generation {gen} =====")
                
                # Run battles for this generation
                battle_results = self.battle_simulator.run_battles(
                    self.strategies,
                    self.strategy_generator,
                    battles_per_generation,
                    tournament_style
                )
                
                # Add to battle history
                self.battle_history.extend(battle_results)
                
                # Evolve population for next generation
                evolved_strategies = self.strategy_generator.evolve_population(
                    self.strategies,
                    battle_results,
                    self.current_generation + 1,  # Next generation number
                    population_size,
                    elite_count=self.config.get("elite_count", 2),
                    tournament_size=self.config.get("tournament_size", 3),
                    min_games_for_evolution=self.config.get("min_games_for_evolution", 3)
                )
                
                # Update our strategies with the evolved ones
                self.strategies = evolved_strategies
                
                # Update metrics
                self._update_metrics()
                
                # Save strategies and metrics
                self._save_best_strategies()
                
                # Create checkpoint if needed
                if (gen + 1) % checkpoint_interval == 0:
                    self._create_checkpoint()
                
                # Log generation time
                generation_time = time.time() - generation_start_time
                self.metrics_history["generation_times"].append(generation_time)
                logger.info(f"Generation {gen} completed in {generation_time:.2f} seconds")
                
                # Visualize progress
                if self.config.get("plot_progress", True) and (gen + 1) % self.config.get("plot_interval", 5) == 0:
                    self._plot_training_progress()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted. Saving checkpoint...")
            self._create_checkpoint()
        
        logger.info("Training completed")
        
        # Return the best strategies found
        return self._get_best_strategies()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load training configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "population_size": 30,
            "generations": 50,
            "battles_per_generation": 50,
            "tournament_style": False,
            "mutation_rate": 0.25,
            "elite_count": 2,
            "tournament_size": 3,
            "min_games_for_evolution": 3,
            "checkpoint_interval": 1,
            "visualize_battles": False,
            "plot_progress": True,
            "plot_interval": 5,
            "data_directory": "training_data",
            "game_executable": "./battle_simulator",
            "default_template": "adaptive"
        }
        
        # Try to load config file
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Merge with default config
                config = {**default_config, **loaded_config}
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
        
        # If file doesn't exist or there was an error, create it with default config
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default configuration at {config_path}")
        except Exception as e:
            logger.warning(f"Error creating default config at {config_path}: {e}")
        
        return default_config
    
    def _update_metrics(self):
        """Update training metrics based on current strategies."""
        if not self.strategies:
            return
        
        # Calculate average and best win rates
        win_rates = []
        best_win_rate = 0
        
        for strategy in self.strategies.values():
            metrics = strategy.get("metrics", {})
            games_played = metrics.get("games_played", 0)
            
            if games_played > 0:
                win_rate = metrics.get("win_rate", 0)
                win_rates.append(win_rate)
                best_win_rate = max(best_win_rate, win_rate)
        
        if win_rates:
            avg_win_rate = sum(win_rates) / len(win_rates)
            self.metrics_history["avg_win_rate"].append(avg_win_rate)
            self.metrics_history["best_win_rate"].append(best_win_rate)
            
            # Calculate diversity (standard deviation of win rates)
            if len(win_rates) > 1:
                diversity = np.std(win_rates)
            else:
                diversity = 0
                
            self.metrics_history["diversity"].append(diversity)
            
            logger.info(f"Metrics - Avg Win Rate: {avg_win_rate:.4f}, Best Win Rate: {best_win_rate:.4f}, Diversity: {diversity:.4f}")
    
    def _save_best_strategies(self):
        """Save the best strategies to the best strategies directory."""
        if not self.strategies:
            return
        
        # Get the top 3 strategies by win rate
        strategies_with_games = [
            (strategy_id, strategy) for strategy_id, strategy in self.strategies.items()
            if strategy.get("metrics", {}).get("games_played", 0) >= self.config.get("min_games_for_evolution", 3)
        ]
        
        if not strategies_with_games:
            logger.warning("No strategies with enough games to save as best strategies")
            return
        
        top_strategies = sorted(
            strategies_with_games,
            key=lambda x: x[1].get("metrics", {}).get("win_rate", 0),
            reverse=True
        )[:3]
        
        # Save each top strategy
        for i, (strategy_id, strategy) in enumerate(top_strategies):
            strategy_code = self.strategy_generator.generate_strategy_code(strategy)
            rank = i + 1
            
            # Create filename with generation and rank
            filename = f"gen_{self.current_generation}_rank_{rank}_{strategy_id}.py"
            filepath = os.path.join(self.best_strategies_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    f.write(strategy_code)
                
                logger.debug(f"Saved rank {rank} strategy to {filepath}")
                
                # Also save the strategy parameters
                params_path = os.path.join(self.best_strategies_dir, f"gen_{self.current_generation}_rank_{rank}_{strategy_id}.json")
                with open(params_path, 'w') as f:
                    json.dump(strategy, f, indent=4)
                
            except Exception as e:
                logger.error(f"Error saving best strategy: {e}")
    
    def _create_checkpoint(self):
        """Create a checkpoint of the current training state."""
        checkpoint_data = {
            "generation": self.current_generation,
            "strategies": self.strategies,
            "battle_history": self.battle_history,
            "metrics_history": self.metrics_history,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create checkpoint filename with timestamp
        timestamp = int(time.time())
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{self.current_generation}_{timestamp}.json")
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"Created checkpoint at {checkpoint_path}")
            
            # Keep only the latest N checkpoints to save space
            max_checkpoints = self.config.get("max_checkpoints", 5)
            self._clean_old_checkpoints(max_checkpoints)
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
    
    def _clean_old_checkpoints(self, max_checkpoints: int):
        """Remove old checkpoints, keeping only the latest N."""
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                filepath = os.path.join(self.checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Remove excess checkpoints
        for filepath, _ in checkpoints[max_checkpoints:]:
            try:
                os.remove(filepath)
                logger.debug(f"Removed old checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"Error removing old checkpoint {filepath}: {e}")
    
    def _load_latest_checkpoint(self):
        """Load the latest checkpoint file if available."""
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                filepath = os.path.join(self.checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getmtime(filepath)))
        
        if not checkpoints:
            logger.info("No checkpoints found")
            return
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        latest_checkpoint_path = checkpoints[0][0]
        
        try:
            with open(latest_checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.current_generation = checkpoint_data.get("generation", 0)
            self.strategies = checkpoint_data.get("strategies", {})
            self.battle_history = checkpoint_data.get("battle_history", [])
            self.metrics_history = checkpoint_data.get("metrics_history", {
                "avg_win_rate": [],
                "best_win_rate": [],
                "diversity": [],
                "generation_times": []
            })
            
            logger.info(f"Resumed from checkpoint: {latest_checkpoint_path} (Generation {self.current_generation})")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {latest_checkpoint_path}: {e}")
    
    def _get_best_strategies(self) -> Dict[str, Any]:
        """Get the best strategies based on win rate."""
        if not self.strategies:
            return {}
        
        # Get strategies with enough games played
        strategies_with_games = {
            strategy_id: strategy for strategy_id, strategy in self.strategies.items()
            if strategy.get("metrics", {}).get("games_played", 0) >= self.config.get("min_games_for_evolution", 3)
        }
        
        if not strategies_with_games:
            return {}
        
        # Sort by win rate
        sorted_strategies = sorted(
            strategies_with_games.items(),
            key=lambda x: x[1].get("metrics", {}).get("win_rate", 0),
            reverse=True
        )
        
        # Return top N strategies
        top_n = min(5, len(sorted_strategies))
        return {strategy_id: strategy for strategy_id, strategy in sorted_strategies[:top_n]}
    
    def _plot_training_progress(self):
        """Create and save plots showing training progress."""
        try:
            # Create figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot win rates
            if self.metrics_history["avg_win_rate"]:
                generations = list(range(len(self.metrics_history["avg_win_rate"])))
                
                # Average and best win rates
                axs[0, 0].plot(generations, self.metrics_history["avg_win_rate"], label="Average Win Rate")
                axs[0, 0].plot(generations, self.metrics_history["best_win_rate"], label="Best Win Rate")
                axs[0, 0].set_xlabel("Generation")
                axs[0, 0].set_ylabel("Win Rate")
                axs[0, 0].set_title("Win Rate Progression")
                axs[0, 0].legend()
                axs[0, 0].grid(True)
                
                # Population diversity
                axs[0, 1].plot(generations, self.metrics_history["diversity"], color='green')
                axs[0, 1].set_xlabel("Generation")
                axs[0, 1].set_ylabel("Standard Deviation of Win Rates")
                axs[0, 1].set_title("Population Diversity")
                axs[0, 1].grid(True)
                
                # Generation times
                if self.metrics_history["generation_times"]:
                    gen_times = self.metrics_history["generation_times"]
                    # Get generations for which we have times
                    time_gens = list(range(len(gen_times)))
                    axs[1, 0].bar(time_gens, gen_times, color='purple')
                    axs[1, 0].set_xlabel("Generation")
                    axs[1, 0].set_ylabel("Time (seconds)")
                    axs[1, 0].set_title("Generation Time")
                    axs[1, 0].grid(True)
                
                # Template type distribution in current population
                if self.strategies:
                    template_counts = {}
                    for strategy in self.strategies.values():
                        template = strategy.get("template_type", "unknown")
                        template_counts[template] = template_counts.get(template, 0) + 1
                    
                    labels = list(template_counts.keys())
                    values = list(template_counts.values())
                    
                    if labels and values:
                        axs[1, 1].pie(values, labels=labels, autopct='%1.1f%%')
                        axs[1, 1].set_title("Strategy Template Distribution")
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"training_progress_gen_{self.current_generation}.png"))
            plt.close()
            
            logger.info(f"Saved training progress plot for generation {self.current_generation}")
            
        except Exception as e:
            logger.error(f"Error creating progress plot: {e}")
            
    def export_best_strategy(self, output_dir: str = None) -> Optional[str]:
        """
        Export the best strategy found during training.
        
        Args:
            output_dir: Directory to export to (defaults to teams directory)
            
        Returns:
            Path to exported strategy file or None if no good strategy found
        """
        best_strategies = self._get_best_strategies()
        
        if not best_strategies:
            logger.warning("No best strategies to export")
            return None
        
        # Get the very best strategy
        best_id, best_strategy = next(iter(best_strategies.items()))
        
        # Generate code for the best strategy
        strategy_code = self.strategy_generator.generate_strategy_code(best_strategy)
        
        # Determine output directory
        if output_dir is None:
            output_dir = self.config.get("teams_directory", "teams")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a clean filename
        filename = f"best_strategy_{best_id}.py"
        output_path = os.path.join(output_dir, filename)
        
        try:
            with open(output_path, 'w') as f:
                f.write(strategy_code)
            
            logger.info(f"Exported best strategy to {output_path}")
            
            # Also save the parameters
            params_path = os.path.join(output_dir, f"best_strategy_{best_id}.json")
            with open(params_path, 'w') as f:
                json.dump(best_strategy, f, indent=4)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting best strategy: {e}")
            return None