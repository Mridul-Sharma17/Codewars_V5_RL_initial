"""
Tower Defense RL - Main Training Script
"""

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime

from rl_config import *
from strategy_generator import StrategyGenerator
from battle_simulator import BattleSimulator
from battle_analyzer import BattleAnalyzer
from checkpoint_manager import CheckpointManager
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'training.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tower Defense RL Training')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--generations', type=int, default=MAX_GENERATIONS, 
                        help='Number of generations to run')
    parser.add_argument('--population', type=int, default=POPULATION_SIZE, 
                        help='Population size')
    return parser.parse_args()

def create_initial_population(population_size):
    """Create an initial population of strategies."""
    logger.info(f"Creating initial population of {population_size} strategies")
    strategies = {}
    
    # Add baseline strategies first
    for baseline in BASELINE_STRATEGIES:
        strategies[baseline["id"]] = baseline
    
    # Generate the rest of the population
    strategy_generator = StrategyGenerator()
    remaining_count = population_size - len(strategies)
    
    if remaining_count > 0:
        generated_strategies = strategy_generator.generate_initial_strategies(remaining_count)
        strategies.update(generated_strategies)
    
    return strategies

def main():
    """Main training loop."""
    args = parse_arguments()
    
    # Initialize components
    checkpoint_manager = CheckpointManager()
    battle_simulator = BattleSimulator()
    battle_analyzer = BattleAnalyzer()
    report_generator = ReportGenerator()
    strategy_generator = StrategyGenerator()
    
    # Load or create initial population
    if args.resume:
        logger.info("Attempting to resume from checkpoint")
        checkpoint = checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            strategies = checkpoint["strategies"]
            metrics = checkpoint["battle_metrics"]
            current_generation = checkpoint["generation"] + 1
            logger.info(f"Resumed from generation {current_generation - 1}")
        else:
            logger.warning("No checkpoint found. Starting from scratch.")
            strategies = create_initial_population(args.population)
            metrics = {}
            current_generation = 0
    else:
        strategies = create_initial_population(args.population)
        metrics = {}
        current_generation = 0
    
    # Main evolution loop
    for generation in range(current_generation, args.generations):
        start_time = time.time()
        logger.info(f"Starting generation {generation}")
        
        # Determine evolution phase and parameters
        if generation <= EXPLORATION_PHASE_END:
            phase = "exploration"
            mutation_rate = INITIAL_MUTATION_RATE
            retention_rate = 1.0  # Keep all strategies
        elif generation <= REFINEMENT_PHASE_END:
            phase = "refinement"
            mutation_rate = REFINEMENT_MUTATION_RATE
            retention_rate = 0.7  # Keep top 70%
        else:
            phase = "specialization"
            mutation_rate = SPECIALIZATION_MUTATION_RATE
            retention_rate = 0.6  # Keep top 60%
        
        logger.info(f"Current phase: {phase}, mutation rate: {mutation_rate}")
        
        # Run round-robin tournament
        battle_results = battle_simulator.run_tournament(strategies)
        metrics.update(battle_results)
        
        # Analyze results and calculate fitness
        fitness_scores = battle_analyzer.calculate_fitness(
            strategies, metrics, generation
        )
        
        # Generate new population
        strategies = strategy_generator.evolve_population(
            strategies, 
            fitness_scores,
            generation, 
            args.population,
            elite_count=ELITE_COUNT,
            tournament_size=TOURNAMENT_SIZE,
            crossover_probability=CROSSOVER_PROBABILITY,
            mutation_rate=mutation_rate,
            retention_rate=retention_rate
        )
        
        # Save checkpoint
        if generation % CHECKPOINT_FREQUENCY == 0:
            checkpoint_manager.save_checkpoint({
                "generation": generation,
                "strategies": strategies,
                "battle_metrics": metrics,
                "fitness_scores": fitness_scores,
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate reports
        report_generator.generate_generation_report(
            generation, strategies, metrics, fitness_scores
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generation {generation} completed in {elapsed_time:.2f} seconds")
    
    # Generate final report
    report_generator.generate_final_report(strategies, metrics)
    
    # Return the best strategy for final submission
    best_strategy = battle_analyzer.get_best_strategy(strategies, metrics)
    logger.info(f"Training complete. Best strategy: {best_strategy['name']} (ID: {best_strategy['id']})")
    
    return best_strategy

if __name__ == "__main__":
    best_strategy = main()
    print(f"Best strategy ID: {best_strategy['id']}")
    print(f"Best strategy file: {best_strategy['source_file']}")