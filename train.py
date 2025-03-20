#!/usr/bin/env python3
"""
Tower Defense Reinforcement Learning System - Training Script
---------------------------------------------------------
Main entry point for training reinforcement learning tower defense strategies.
Provides command-line interface for controlling the training process.

Created: 2025-03-20
Author: Mridul-Sharma17
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime

from training_manager import TrainingManager
from strategy_generator import StrategyGenerator
from battle_simulator import BattleSimulator

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_data/train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainScript")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train tower defense strategies using reinforcement learning.")
    
    # Basic training parameters
    parser.add_argument("-g", "--generations", type=int, default=None,
                        help="Number of generations to train")
    parser.add_argument("-p", "--population", type=int, default=None,
                        help="Size of the strategy population")
    parser.add_argument("-b", "--battles", type=int, default=None,
                        help="Number of battles per generation")
    
    # Training control
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from checkpoint, start fresh")
    parser.add_argument("--config", type=str, default="training_config.json",
                        help="Path to training configuration file")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Visualize battles instead of running headless")
    
    # Output control
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Directory to export best strategies to")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots after training")
    
    # Advanced options
    parser.add_argument("--tournament", action="store_true",
                        help="Use tournament-style battles")
    parser.add_argument("--mutation-rate", type=float, default=None,
                        help="Set mutation rate for genetic algorithm")
    parser.add_argument("--elite-count", type=int, default=None,
                        help="Number of elite strategies to keep unchanged")
    
    # Special operations
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze training results without training")
    parser.add_argument("--battle-test", type=str, nargs=2, metavar=("STRATEGY1", "STRATEGY2"),
                        help="Run a single battle between two strategy files")
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update configuration with command-line arguments."""
    # Only update config values that were explicitly specified in args
    if args.generations is not None:
        config["generations"] = args.generations
    
    if args.population is not None:
        config["population_size"] = args.population
    
    if args.battles is not None:
        config["battles_per_generation"] = args.battles
    
    if args.tournament:
        config["tournament_style"] = True
    
    if args.mutation_rate is not None:
        config["mutation_rate"] = args.mutation_rate
    
    if args.elite_count is not None:
        config["elite_count"] = args.elite_count
    
    if args.visualize:
        config["visualize_battles"] = True
    
    if args.plot:
        config["plot_progress"] = True
    
    return config

def save_config(config, config_path):
    """Save updated configuration back to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Updated configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def run_battle_test(strategy1_path, strategy2_path):
    """Run a single battle between two strategy files for testing."""
    # Create a temporary directory for battle test
    test_dir = "battle_test_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy strategy files to the temp directory
    import shutil
    strat1_name = os.path.basename(strategy1_path)
    strat2_name = os.path.basename(strategy2_path)
    
    temp_strat1 = os.path.join(test_dir, strat1_name)
    temp_strat2 = os.path.join(test_dir, strat2_name)
    
    shutil.copy(strategy1_path, temp_strat1)
    shutil.copy(strategy2_path, temp_strat2)
    
    # Create a battle simulator
    simulator = BattleSimulator(game_executable="./battle_simulator")
    simulator.temp_dir = test_dir
    simulator.enable_visualization(True)
    
    logger.info(f"Running battle test: {strat1_name} vs {strat2_name}")
    
    # Run the battle
    result = simulator._run_battle(strat1_name.replace(".py", ""), strat2_name.replace(".py", ""))
    
    if result:
        logger.info(f"Battle result: {result['winner']} defeated {result['loser']} "
                   f"with {result['winner_tower_health']} health remaining")
        logger.info(f"Game duration: {result['game_duration']} frames")
    else:
        logger.error("Battle test failed")
    
    # Clean up
    try:
        shutil.rmtree(test_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory: {e}")
    
    return result

def analyze_training_results(training_manager):
    """Analyze and display training results without running training."""
    logger.info("Analyzing training results...")
    
    # Get metrics history
    metrics = training_manager.metrics_history
    
    if not metrics.get("avg_win_rate"):
        logger.warning("No metrics data available to analyze")
        return
    
    # Print basic stats
    generations = len(metrics["avg_win_rate"])
    logger.info(f"Training data for {generations} generations")
    
    if generations > 0:
        # Win rate stats
        avg_win_rates = metrics["avg_win_rate"]
        best_win_rates = metrics["best_win_rate"]
        
        logger.info(f"Final average win rate: {avg_win_rates[-1]:.4f}")
        logger.info(f"Final best win rate: {best_win_rates[-1]:.4f}")
        logger.info(f"Best win rate ever: {max(best_win_rates):.4f} (Generation {best_win_rates.index(max(best_win_rates))})")
        
        # Diversity stats
        if metrics.get("diversity"):
            diversity = metrics["diversity"]
            logger.info(f"Final population diversity: {diversity[-1]:.4f}")
            
        # Performance stats
        if metrics.get("generation_times"):
            gen_times = metrics["generation_times"]
            avg_time = sum(gen_times) / len(gen_times)
            logger.info(f"Average generation time: {avg_time:.2f} seconds")
        
        # Get best strategies
        best_strategies = training_manager._get_best_strategies()
        logger.info(f"Top {len(best_strategies)} strategies:")
        
        for i, (strategy_id, strategy) in enumerate(best_strategies.items(), 1):
            metrics = strategy.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            games = metrics.get("games_played", 0)
            template = strategy.get("template_type", "unknown")
            
            logger.info(f"{i}. Strategy {strategy_id}: Win Rate {win_rate:.4f} ({games} games) - Template: {template}")
        
        # Generate plots
        training_manager._plot_training_progress()
        logger.info("Generated training progress plot")

def main():
    """Main entry point for training script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Special case: Battle test mode
    if args.battle_test:
        strategy1_path, strategy2_path = args.battle_test
        
        if not os.path.exists(strategy1_path) or not os.path.exists(strategy2_path):
            logger.error("One or both strategy files do not exist")
            return 1
        
        run_battle_test(strategy1_path, strategy2_path)
        return 0
    
    # Create output directories
    os.makedirs("training_data", exist_ok=True)
    
    # Load the training manager
    training_manager = TrainingManager(
        config_path=args.config,
        resume_from_checkpoint=not args.no_resume
    )
    
    # Get current config
    config = training_manager.config
    
    # Update config from command line args
    config = update_config_from_args(config, args)
    
    # Save updated config
    save_config(config, args.config)
    
    # Analysis-only mode
    if args.analyze:
        analyze_training_results(training_manager)
        return 0
    
    # Run training
    logger.info("Starting training process...")
    start_time = time.time()
    
    try:
        # Run training
        training_manager.train(generations=config.get("generations"))
        
        # Export best strategy
        export_dir = args.export_dir or config.get("teams_directory", "teams")
        exported_path = training_manager.export_best_strategy(export_dir)
        
        if exported_path:
            logger.info(f"Exported best strategy to {exported_path}")
        
        # Generate final plots
        if config.get("plot_progress", True):
            training_manager._plot_training_progress()
            logger.info("Generated final training progress plot")
        
        # Log completion statistics
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Trained for {training_manager.current_generation + 1} generations")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Saving checkpoint...")
        training_manager._create_checkpoint()
        
        logger.info("You can resume training by running this script again")
        return 1
    
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        logger.info("Attempting to save checkpoint...")
        try:
            training_manager._create_checkpoint()
            logger.info("Checkpoint saved")
        except:
            logger.error("Failed to save checkpoint", exc_info=True)
        
        return 2

if __name__ == "__main__":
    sys.exit(main())