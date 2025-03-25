#!/usr/bin/env python3

import os
import sys
import argparse
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import random
import matplotlib.pyplot as plt

# Import our modules
from rl_trainer import RLTrainer
from game_runner import GameRunner 
from strategy_generator import StrategyGenerator
from evolve import StrategyEvolver
from battle_analyzer import BattleAnalyzer
from strategy_encoder import StrategyEncoder
from improved_strategy_selection import select_best_strategies, calculate_wilson_score

class TrainingManager:
    """
    Main class that orchestrates the entire reinforcement learning training process.
    """
    
    def __init__(self, output_dir="training_data", headless=False):
        """
        Initialize the training manager.
        
        Args:
            output_dir: Directory for storing training data and results
            headless: Whether to run games without graphics (auto-closes windows)
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing training components...")
        self.trainer = RLTrainer(output_dir=output_dir)
        self.game_runner = GameRunner(game_path="game", headless=headless)
        self.strategy_generator = StrategyGenerator()
        self.evolver = StrategyEvolver()
        self.analyzer = BattleAnalyzer(metrics_file=os.path.join(output_dir, "battle_metrics.json"),
                                     visualization_dir=os.path.join(output_dir, "visualizations"))
        self.encoder = StrategyEncoder()
        
        # Training configuration - default values
        self.config = {
            "generations": 10,
            "battles_per_generation": 20,
            "strategies_per_generation": 4,
            "min_games_for_evolution": 3,
            "tournament_size": 3,
            "max_strategies": 10  # Maximum number of strategies to maintain
        }
        
        # Load config if exists
        config_path = os.path.join(output_dir, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                self.config.update(saved_config)
                print(f"Configuration loaded from {config_path}: {self.config}")
        else:
            print(f"Configuration file {config_path} not found. Using default values.")
        
        # Create teams directory structure
        self.teams_dir = os.path.join(output_dir, "teams")
        Path(self.teams_dir).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_generation = 0
        self.best_strategy_id = None
        self.training_start_time = None
        
        # Track generation champions
        self.generation_champions = {}
        
        # Track per-generation metrics
        self.generation_metrics = {}
        
        # Load training state if exists
        state_path = os.path.join(output_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.current_generation = state.get("current_generation", 0)
                self.best_strategy_id = state.get("best_strategy_id", None)
                self.generation_champions = state.get("generation_champions", {})
                self.generation_metrics = state.get("generation_metrics", {})
    
    def save_config(self):
        """Save training configuration."""
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def save_state(self):
        """Save current training state."""
        state_path = os.path.join(self.output_dir, "training_state.json")
        state = {
            "current_generation": self.current_generation,
            "best_strategy_id": self.best_strategy_id,
            "generation_champions": self.generation_champions,
            "generation_metrics": self.generation_metrics,
            "timestamp": datetime.now().isoformat()
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def generate_team_files(self):
        """Generate Python files for all strategies."""
        teams_dir = os.path.join(self.teams_dir, f"gen_{self.current_generation}")
        Path(teams_dir).mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        for strategy_id, strategy in self.trainer.strategies.items():
            file_path = os.path.join(teams_dir, f"{strategy_id}.py")
            self.strategy_generator.generate_code(strategy, file_path)
            file_paths.append(file_path)
            
        return file_paths
    
    def select_battle_pairs(self, team_files):
        """
        Select pairs of strategies to create a complete round-robin tournament.
        Each strategy battles every other strategy as both team A and team B.
        
        Args:
            team_files: List of team Python files
            
        Returns:
            List of (team1_file, team2_file) pairs to battle
        """
        pairs = []
        
        # Create round-robin tournament pairs
        for i, team1_file in enumerate(team_files):
            for j, team2_file in enumerate(team_files):
                if i != j:  # Don't battle against self
                    # Add both A vs B configurations
                    pairs.append((team1_file, team2_file))
        
        # If we have too many pairs for the configured battles_per_generation,
        # select a balanced subset, preserving position balance
        battles_per_gen = self.config["battles_per_generation"]
        if len(pairs) > battles_per_gen:
            # Track how many times each file appears
            file_counts = {}
            for team1, team2 in pairs:
                file_counts[team1] = file_counts.get(team1, 0) + 1
                file_counts[team2] = file_counts.get(team2, 0) + 1
            
            # Sort pairs to prioritize those involving strategies with fewer battles
            pairs.sort(key=lambda pair: file_counts[pair[0]] + file_counts[pair[1]])
            
            # Take top battles_per_gen pairs
            pairs = pairs[:battles_per_gen]
        
        # Shuffle the final list to avoid systemic bias
        random.shuffle(pairs)
        
        return pairs
    
    def run_generation_battles(self, team_files):
        """
        Run battles between strategies for the current generation.
        
        Args:
            team_files: List of team Python files
            
        Returns:
            List of battle results
        """
        print(f"\n=== Running Generation {self.current_generation} Battles ===")
        print(f"Number of strategies: {len(team_files)}")
        
        # Create pairs of strategies to battle using round-robin tournament structure
        battle_pairs = self.select_battle_pairs(team_files)
        print(f"Battles to run: {len(battle_pairs)}")
        
        # Initialize current generation metrics
        gen_key = str(self.current_generation)
        self.generation_metrics[gen_key] = {}
        
        # Run all battles
        all_results = []
        for i, (team1, team2) in enumerate(battle_pairs):
            print(f"\nBattle {i+1}/{len(battle_pairs)}: {os.path.basename(team1)} vs {os.path.basename(team2)}")
            
            try:
                result = self.game_runner.run_battle(team1, team2)
                if result:
                    # Add result to analyzer
                    self.analyzer.add_battle_result(result)
                    all_results.append(result)
                    
                    # Update global strategy metrics
                    self._update_strategy_metrics(team1, team2, result)
                    
                    # Update per-generation metrics
                    self._update_generation_metrics(team1, team2, result)
            except Exception as e:
                print(f"Error running battle: {e}")
        
        # Save strategies with updated metrics
        self.trainer.save_strategies()
        
        return all_results
    
    def _update_strategy_metrics(self, team1_file, team2_file, result):
        """
        Update strategy metrics based on battle result.
        
        Args:
            team1_file: Path to team 1's file
            team2_file: Path to team 2's file
            result: Battle result dictionary
        """
        # Extract strategy IDs from filenames
        team1_id = os.path.basename(team1_file).split('.')[0]
        team2_id = os.path.basename(team2_file).split('.')[0]
        
        if team1_id in self.trainer.strategies and team2_id in self.trainer.strategies:
            # Update games played
            self.trainer.strategies[team1_id]["metrics"]["games_played"] += 1
            self.trainer.strategies[team2_id]["metrics"]["games_played"] += 1
            
            # Update wins/losses
            if result.get("winner") == result.get("team1_name"):
                self.trainer.strategies[team1_id]["metrics"]["wins"] += 1
                self.trainer.strategies[team2_id]["metrics"]["losses"] += 1
            elif result.get("winner") == result.get("team2_name"):
                self.trainer.strategies[team2_id]["metrics"]["wins"] += 1
                self.trainer.strategies[team1_id]["metrics"]["losses"] += 1
            
            # Update win rates and Wilson scores
            for strategy_id in [team1_id, team2_id]:
                metrics = self.trainer.strategies[strategy_id]["metrics"]
                if metrics["games_played"] > 0:
                    metrics["win_rate"] = metrics["wins"] / metrics["games_played"]
                    if metrics["games_played"] >= 3:
                        metrics["wilson_score"] = calculate_wilson_score(
                            metrics["wins"], metrics["games_played"]
                        )
    
    def _update_generation_metrics(self, team1_file, team2_file, result):
        """
        Update per-generation metrics based on battle result.
        
        Args:
            team1_file: Path to team 1's file
            team2_file: Path to team 2's file
            result: Battle result dictionary
        """
        # Extract strategy IDs from filenames
        team1_id = os.path.basename(team1_file).split('.')[0]
        team2_id = os.path.basename(team2_file).split('.')[0]
        
        gen_key = str(self.current_generation)
        
        # Initialize metrics for these strategies if not exist
        for strategy_id in [team1_id, team2_id]:
            if strategy_id not in self.generation_metrics[gen_key]:
                self.generation_metrics[gen_key][strategy_id] = {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0
                }
        
        # Update games played
        self.generation_metrics[gen_key][team1_id]["games"] += 1
        self.generation_metrics[gen_key][team2_id]["games"] += 1
        
        # Update wins/losses
        if result.get("winner") == result.get("team1_name"):
            self.generation_metrics[gen_key][team1_id]["wins"] += 1
            self.generation_metrics[gen_key][team2_id]["losses"] += 1
        elif result.get("winner") == result.get("team2_name"):
            self.generation_metrics[gen_key][team2_id]["wins"] += 1
            self.generation_metrics[gen_key][team1_id]["losses"] += 1
        
        # Update win rates
        for strategy_id in [team1_id, team2_id]:
            metrics = self.generation_metrics[gen_key][strategy_id]
            if metrics["games"] > 0:
                metrics["win_rate"] = metrics["wins"] / metrics["games"]
    
    def evolve_strategies(self):
        """Evolve new strategies based on battle performance."""
        print("\n=== Evolving New Strategies ===")
        
        # Find strategies with enough games played
        eligible_strategies = {
            strategy_id: strategy
            for strategy_id, strategy in self.trainer.strategies.items()
            if strategy["metrics"]["games_played"] >= self.config["min_games_for_evolution"]
        }
        
        if not eligible_strategies:
            print("No strategies have enough games to evolve from yet.")
            return []
        
        # Get number of strategies to create
        num_new_strategies = self.config["strategies_per_generation"]
        
        # Generate evolved strategies with adaptive capabilities if available
        if hasattr(self.evolver, 'evolve_population_with_adaptive'):
            new_strategies = self.evolver.evolve_population_with_adaptive(
                eligible_strategies, 
                num_offspring=num_new_strategies
            )
        else:
            # Use standard evolution otherwise
            new_strategies = self.evolver.evolve_population(
                eligible_strategies, 
                num_offspring=num_new_strategies
            )
        
        # Add new strategies to trainer
        for strategy in new_strategies:
            self.trainer.strategies[strategy["id"]] = strategy
        
        # Save updated strategies
        self.trainer.save_strategies()
        
        print(f"Added {len(new_strategies)} new evolved strategies.")
        return new_strategies
    
    def prune_strategies(self, new_strategies):
        """
        Prune strategies to maintain max_strategies limit based on current generation performance.
        
        Args:
            new_strategies: List of newly added strategies (these will not be pruned)
        """
        max_strategies = self.config["max_strategies"]
        
        # If we're under the limit, no need to prune
        if len(self.trainer.strategies) <= max_strategies:
            return
        
        print(f"\n=== Pruning Strategies (Current: {len(self.trainer.strategies)}, Max: {max_strategies}) ===")
        
        # Get the current generation key
        gen_key = str(self.current_generation)
        
        # Collect strategies with their performance data
        strategy_performances = []
        new_strategy_ids = [s["id"] for s in new_strategies]
        
        for strategy_id, strategy in self.trainer.strategies.items():
            # Skip new strategies that haven't played a game yet
            if strategy_id in new_strategy_ids:
                print(f"Preserving new strategy: {strategy_id}")
                continue
                
            # Get current generation performance if available
            gen_win_rate = 0.0
            if gen_key in self.generation_metrics and strategy_id in self.generation_metrics[gen_key]:
                metrics = self.generation_metrics[gen_key][strategy_id]
                if metrics["games"] > 0:
                    gen_win_rate = metrics["win_rate"]
            
            # Use a tuple to store (strategy_id, generation win rate, games played)
            strategy_performances.append((strategy_id, gen_win_rate, 
                                         self.generation_metrics[gen_key].get(strategy_id, {}).get("games", 0)))
        
        # Sort by win rate (ascending) and then by games played (ascending)
        # This sorts the worst performers first
        strategy_performances.sort(key=lambda x: (x[1], x[2]))
        
        # Calculate how many to prune
        num_to_keep = max_strategies - len(new_strategy_ids)
        num_to_prune = max(0, len(strategy_performances) - num_to_keep)
        
        if num_to_prune <= 0:
            print("No pruning needed.")
            return
        
        # Identify strategies to prune
        strategies_to_prune = strategy_performances[:num_to_prune]
        
        # Print pruning information
        print(f"Pruning {num_to_prune} strategies:")
        for strategy_id, win_rate, games in strategies_to_prune:
            print(f"- {strategy_id}: Win rate {win_rate:.2f} ({games} games)")
            
            # Remove from trainer strategies
            if strategy_id in self.trainer.strategies:
                del self.trainer.strategies[strategy_id]
        
        # Save updated strategies
        self.trainer.save_strategies()
        print(f"After pruning: {len(self.trainer.strategies)} strategies remain")
    
    def find_generation_champion(self):
        """
        Find the best performing strategy for the current generation only.
        
        Returns:
            ID of the generation champion
        """
        gen_key = str(self.current_generation)
        
        # Check if we have metrics for this generation
        if gen_key not in self.generation_metrics or not self.generation_metrics[gen_key]:
            print(f"No metrics available for generation {self.current_generation}")
            return None
        
        # Find strategies with at least 3 games
        qualified_strategies = []
        for strategy_id, metrics in self.generation_metrics[gen_key].items():
            if metrics["games"] >= 3:
                qualified_strategies.append((strategy_id, metrics))
        
        if not qualified_strategies:
            print(f"No strategies with enough games in generation {self.current_generation}")
            return None
        
        # Sort by win rate
        sorted_strategies = sorted(
            qualified_strategies,
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )
        
        # The champion is the strategy with highest win rate
        champion_id = sorted_strategies[0][0]
        champion_metrics = sorted_strategies[0][1]
        
        # Store the champion for this generation
        self.generation_champions[gen_key] = {
            "id": champion_id,
            "win_rate": champion_metrics["win_rate"],
            "wins": champion_metrics["wins"],
            "games": champion_metrics["games"]
        }
        
        print(f"\nGeneration {self.current_generation} Champion: {champion_id}")
        print(f"Win rate: {champion_metrics['win_rate']:.2f} ({champion_metrics['wins']}/{champion_metrics['games']} games)")
        
        return champion_id
    
    def select_championship_contenders(self, top_n=5):
        """
        Select top contenders for championship from all generation champions.
        
        Args:
            top_n: Number of top strategies to include
            
        Returns:
            List of (strategy_id, strategy) tuples
        """
        # First, get all generation champions 
        champions = []
        for gen, champion_data in self.generation_champions.items():
            strategy_id = champion_data["id"]
            if strategy_id in self.trainer.strategies:
                champions.append((strategy_id, self.trainer.strategies[strategy_id]))
        
        # If we don't have enough champions, add some top performers from the last generation
        if len(champions) < top_n:
            last_gen_key = str(self.current_generation)
            if last_gen_key in self.generation_metrics:
                # Get strategies from last generation
                last_gen_strategies = []
                for strategy_id, metrics in self.generation_metrics[last_gen_key].items():
                    # Skip the champion we already added
                    if last_gen_key in self.generation_champions and strategy_id == self.generation_champions[last_gen_key]["id"]:
                        continue
                    
                    if metrics["games"] >= 3 and strategy_id in self.trainer.strategies:
                        last_gen_strategies.append((strategy_id, metrics["win_rate"]))
                
                # Sort by win rate and take the best ones
                last_gen_strategies.sort(key=lambda x: x[1], reverse=True)
                
                # Add top strategies until we reach top_n
                remaining_slots = top_n - len(champions)
                for i in range(min(remaining_slots, len(last_gen_strategies))):
                    strategy_id = last_gen_strategies[i][0]
                    champions.append((strategy_id, self.trainer.strategies[strategy_id]))
        
        # If we still don't have enough, add top overall performers
        if len(champions) < top_n:
            # Get all strategies with enough games
            qualified_strategies = []
            for strategy_id, strategy in self.trainer.strategies.items():
                # Skip those we already added
                if any(c[0] == strategy_id for c in champions):
                    continue
                
                if strategy["metrics"]["games_played"] >= 3:
                    qualified_strategies.append((strategy_id, strategy))
            
            # Sort by win rate
            qualified_strategies.sort(
                key=lambda x: x[1]["metrics"]["win_rate"],
                reverse=True
            )
            
            # Add top strategies until we reach top_n
            remaining_slots = top_n - len(champions)
            champions.extend(qualified_strategies[:remaining_slots])
        
        # Make sure we have enough contenders (minimum 2)
        if len(champions) < 2:
            print("Warning: Not enough valid strategies for championship")
            # Fall back to all strategies with enough games
            all_strategies = []
            for strategy_id, strategy in self.trainer.strategies.items():
                if strategy["metrics"]["games_played"] >= 3:
                    all_strategies.append((strategy_id, strategy))
            
            all_strategies.sort(
                key=lambda x: x[1]["metrics"]["win_rate"],
                reverse=True
            )
            champions = all_strategies[:top_n]
        
        return champions
    
    def run_championship(self, top_n=5, battles_per_matchup=2):
        """
        Run a championship tournament to determine the best strategy.
        
        Args:
            top_n: Number of top strategies to include in championship
            battles_per_matchup: Number of battles to run for each strategy pair
            
        Returns:
            ID of the winning strategy
        """
        print("\n=== Running Championship Tournament ===")
        
        # Select top contenders using the specialized method
        top_contenders = self.select_championship_contenders(top_n)
        
        if not top_contenders:
            print("Not enough strategies with sufficient games for championship")
            return None
        
        contender_ids = [id for id, _ in top_contenders]
        print(f"Championship contenders: {', '.join(contender_ids)}")
        
        # Create tournament pairs ensuring each strategy battles every other
        tournament_pairs = []
        for i, (id1, _) in enumerate(top_contenders):
            for j, (id2, _) in enumerate(top_contenders):
                if i != j:  # Don't battle against self
                    for _ in range(battles_per_matchup):
                        tournament_pairs.append((id1, id2))
        
        # Shuffle to avoid systematic bias
        random.shuffle(tournament_pairs)
        
        # Run the tournament
        print(f"Running {len(tournament_pairs)} championship battles...")
        championship_results = {id: {"wins": 0, "games": 0} for id, _ in top_contenders}
        
        for team1_id, team2_id in tournament_pairs:
            # Find paths to team files
            team1_path = self._get_strategy_path(team1_id)
            team2_path = self._get_strategy_path(team2_id)
            
            if not team1_path or not team2_path:
                print(f"Warning: Could not find path for {team1_id} or {team2_id}, skipping match")
                continue
            
            print(f"Championship battle: {team1_id} vs {team2_id}")
            result = self.game_runner.run_battle(team1_path, team2_path)
            
            if result:
                # Update championship metrics
                if result["winner"] == result["team1_name"]:
                    championship_results[team1_id]["wins"] += 1
                elif result["winner"] == result["team2_name"]:
                    championship_results[team2_id]["wins"] += 1
                    
                championship_results[team1_id]["games"] += 1
                championship_results[team2_id]["games"] += 1
        
        # Calculate win rates and find champion
        for id in championship_results:
            if championship_results[id]["games"] > 0:
                championship_results[id]["win_rate"] = (
                    championship_results[id]["wins"] / championship_results[id]["games"]
                )
            else:
                championship_results[id]["win_rate"] = 0
        
        # Sort by win rate
        sorted_results = sorted(
            championship_results.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )
        
        # Print championship results
        print("\nChampionship Results:")
        print("--------------------")
        for strategy_id, results in sorted_results:
            print(f"{strategy_id}: Win rate {results['win_rate']:.2f} ({results['wins']}/{results['games']})")
        
        if sorted_results:
            champion_id = sorted_results[0][0]
            print(f"\nChampionship winner: {champion_id}")
            print(f"Win rate: {sorted_results[0][1]['win_rate']:.2f} "
                  f"({sorted_results[0][1]['wins']}/{sorted_results[0][1]['games']})")
            return champion_id
        else:
            print("No championship results available")
            return None
    
    def _get_strategy_path(self, strategy_id):
        """
        Find the file path for a given strategy ID.
        
        Args:
            strategy_id: ID of the strategy to find
            
        Returns:
            Full path to the strategy file, or None if not found
        """
        # Check if this is a generation-based ID
        gen_number = None
        if strategy_id.startswith("gen_"):
            parts = strategy_id.split("_")
            if len(parts) > 1 and parts[1].isdigit():
                gen_number = int(parts[1])
        
        # Determine generations to search
        generations_to_search = []
        if gen_number is not None:
            # If we know the generation, search that one first
            generations_to_search.append(gen_number)
        
        # Add all other generations in descending order (newest first)
        for gen in range(self.current_generation, 0, -1):
            if gen not in generations_to_search:
                generations_to_search.append(gen)
        
        # Also check for non-generation files in the teams directory
        # First, try in generation folders
        for gen in generations_to_search:
            gen_dir = os.path.join(self.teams_dir, f"gen_{gen}")
            if os.path.exists(gen_dir):
                # Try with .py extension
                path = os.path.join(gen_dir, f"{strategy_id}.py")
                if os.path.exists(path):
                    return path
                
                # Try without extension if ID already includes .py
                if strategy_id.endswith(".py"):
                    path = os.path.join(gen_dir, strategy_id)
                    if os.path.exists(path):
                        return path
        
        # Check in main teams directory
        teams_dir = "teams"
        if os.path.exists(teams_dir):
            # Try with .py extension
            path = os.path.join(teams_dir, f"{strategy_id}.py")
            if os.path.exists(path):
                return path
            
            # Try without extension
            if strategy_id.endswith(".py"):
                path = os.path.join(teams_dir, strategy_id)
                if os.path.exists(path):
                    return path
        
        # As a last resort, search all generation folders for partial matches
        for gen in range(self.current_generation, 0, -1):
            gen_dir = os.path.join(self.teams_dir, f"gen_{gen}")
            if os.path.exists(gen_dir):
                for file in os.listdir(gen_dir):
                    if file.startswith(strategy_id) or strategy_id in file:
                        return os.path.join(gen_dir, file)
        
        print(f"Warning: Could not find path for strategy {strategy_id}")
        return None
    
    def find_best_strategy(self):
        """
        Find the best strategy based on the current generation or final tournament.
        Only runs the championship tournament at the final generation.
        
        Returns:
            ID of the best strategy
        """
        # For regular generations, just find the generation champion
        if self.current_generation < self.current_generation + self.config["generations"]:
            # Find generation champion
            champion_id = self.find_generation_champion()
            
            if champion_id:
                # Store as best for now (will be replaced in final generation)
                self.best_strategy_id = champion_id
                return champion_id
            else:
                print("\nNo strategies with enough games to determine best yet.")
                return None
        
        # For the final generation, run the championship tournament
        print("\n=== Final Generation: Running Championship Tournament ===")
        best_strategy_id = self.run_championship(top_n=5, battles_per_matchup=3)
        
        if best_strategy_id:
            # Update best strategy ID
            self.best_strategy_id = best_strategy_id
            
            # Print info about the best strategy
            strategy = self.trainer.strategies.get(best_strategy_id)
            if strategy:
                print(f"\nBest strategy selected: {best_strategy_id}")
                if "metrics" in strategy:
                    metrics = strategy["metrics"]
                    print(f"Overall stats: Win rate {metrics.get('win_rate', 0):.2f} "
                          f"({metrics.get('wins', 0)}/{metrics.get('games_played', 0)} games)")
            
            return best_strategy_id
        else:
            print("\nNo strategies with enough games to determine best yet.")
            return None
    
    def generate_final_submission(self):
        """Generate a final submission file from the best strategy"""
        print("\n=== Generating Final Submission ===")
        
        # Run the championship tournament to find the true best strategy
        final_champion = self.run_championship(top_n=5, battles_per_matchup=3)
        if final_champion:
            self.best_strategy_id = final_champion
        
        if not self.best_strategy_id:
            print("Cannot generate final submission: no best strategy found.")
            return
        
        # Get the correct path for best strategy - look through all generation folders
        strategy_path = None
        all_found_paths = []
        
        # Search through all generation folders
        training_data_dir = "training_data/teams"
        for gen_folder in sorted(os.listdir(training_data_dir), reverse=True):
            if gen_folder.startswith("gen_"):
                potential_path = os.path.join(training_data_dir, gen_folder, f"{self.best_strategy_id}.py")
                if os.path.exists(potential_path):
                    strategy_path = potential_path
                    break
                
                # Also check for files containing the ID
                core_id = self.best_strategy_id
                if core_id.endswith(".py"):
                    core_id = core_id[:-3]
                    
                for file in os.listdir(os.path.join(training_data_dir, gen_folder)):
                    if core_id in file:
                        all_found_paths.append(os.path.join(training_data_dir, gen_folder, file))
        
        # If we didn't find an exact match but found partial matches, use the first one
        if not strategy_path and all_found_paths:
            strategy_path = all_found_paths[0]
        
        # If still no path found, give up
        if not strategy_path:
            print(f"Error: Could not find best strategy file for ID: {self.best_strategy_id}")
            print("Searched in all generation folders in training_data/teams/")
            return
        
        # Read the strategy code
        try:
            with open(strategy_path, 'r') as f:
                strategy_code = f.read()
                print(f"Found best strategy at: {strategy_path}")
        except Exception as e:
            print(f"Error reading best strategy file: {e}")
            return
            
        # Create the submission file
        submission_file = "submission.py"
        try:
            with open(submission_file, 'w') as f:
                f.write(strategy_code)
            print(f"Final submission saved to {submission_file}")
            
            # Make sure the teams directory exists
            teams_dir = "teams"
            if not os.path.exists(teams_dir):
                os.makedirs(teams_dir)
                
            # Add a copy to the teams directory so it can be tested
            shutil.copy(submission_file, os.path.join(teams_dir, "submission.py"))
            print(f"Submission also copied to teams/submission.py for testing")
        except Exception as e:
            print(f"Error saving submission file: {e}")
    
    def generate_training_report(self):
        """Generate a comprehensive training report."""
        print("\n=== Generating Training Report ===")
        
        # Generate visualizations
        self.analyzer.generate_reports()
        
        # Calculate training statistics
        num_strategies = len(self.trainer.strategies)
        total_battles = len(self.analyzer.metrics)
        
        # Find strategies by generation
        strategies_by_gen = {}
        for strategy_id in self.trainer.strategies:
            # Extract generation from ID
            if "_" in strategy_id:
                prefix = strategy_id.split('_')[0]
                if prefix == "gen":
                    # Format: gen_ABCD1234
                    gen_num = self.current_generation
                    if gen_num not in strategies_by_gen:
                        strategies_by_gen[gen_num] = []
                    strategies_by_gen[gen_num].append(strategy_id)
                elif prefix.startswith("gen") and prefix[3:].isdigit():
                    # Format: gen5_ABCD1234
                    gen_num = int(prefix[3:])
                    if gen_num not in strategies_by_gen:
                        strategies_by_gen[gen_num] = []
                    strategies_by_gen[gen_num].append(strategy_id)
        
        # Create a report file
        report_path = os.path.join(self.output_dir, "training_report.md")
        with open(report_path, 'w') as f:
            f.write("# Tower Defense Strategy Training Report\n\n")
            
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training summary
            f.write("## Training Summary\n\n")
            f.write(f"- Total generations: {self.current_generation}\n")
            f.write(f"- Total strategies: {num_strategies}\n")
            f.write(f"- Total battles: {total_battles}\n")
            if self.training_start_time:
                duration = time.time() - self.training_start_time
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                f.write(f"- Training duration: {hours}h {minutes}m\n")
            f.write("\n")
            
            # Best strategy
            f.write("## Best Strategy\n\n")
            if self.best_strategy_id and self.best_strategy_id in self.trainer.strategies:
                best = self.trainer.strategies[self.best_strategy_id]
                f.write(f"- ID: {self.best_strategy_id}\n")
                f.write(f"- Type: {best.get('name', 'Unknown')}\n")
                f.write(f"- Games played: {best['metrics']['games_played']}\n")
                f.write(f"- Win rate: {best['metrics']['win_rate']:.2f} ({best['metrics']['wins']}/{best['metrics']['games_played']})\n")
                wilson = best['metrics'].get('wilson_score', 0)
                if wilson > 0:
                    f.write(f"- Wilson score: {wilson:.4f}\n")
                
                # Strategy parameters
                f.write("\nKey parameters:\n")
                f.write(f"- Lane preference: {best.get('lane_preference', 'Unknown')}\n")
                f.write(f"- Defensive trigger: {best.get('defensive_trigger', 'Unknown')}\n")
                if 'elixir_thresholds' in best:
                    f.write(f"- Min deploy elixir: {best['elixir_thresholds'].get('min_deploy', 'Unknown')}\n")
                if 'position_settings' in best:
                    f.write(f"- X range: {best['position_settings'].get('x_range', 'Unknown')}\n")
                    f.write(f"- Default Y: {best['position_settings'].get('y_default', 'Unknown')}\n")
                if 'troop_priority' in best:
                    f.write(f"- Troop priority: {', '.join(best['troop_priority'][:5])}\n")
            else:
                f.write("No best strategy identified yet.\n")
            f.write("\n")
            
            # Generation champions
            f.write("## Generation Champions\n\n")
            f.write("| Generation | Champion | Win Rate | Games |\n")
            f.write("|------------|----------|----------|-------|\n")
            for gen in sorted(self.generation_champions.keys()):
                champion = self.generation_champions[gen]
                f.write(f"| {gen} | {champion['id']} | {champion['win_rate']:.2f} | {champion['games']} |\n")
            f.write("\n")
            
            # Strategy evolution
            f.write("## Strategy Evolution\n\n")
            f.write("| Generation | Strategies | Avg Win Rate | Best Win Rate | Best Wilson Score |\n")
            f.write("|------------|------------|--------------|---------------|------------------|\n")
            for gen in sorted(strategies_by_gen.keys()):
                strats = strategies_by_gen[gen]
                win_rates = [self.trainer.strategies[s]["metrics"]["win_rate"] 
                           for s in strats 
                           if self.trainer.strategies[s]["metrics"]["games_played"] > 0]
                
                wilson_scores = [self.trainer.strategies[s]["metrics"].get("wilson_score", 0) 
                               for s in strats 
                               if self.trainer.strategies[s]["metrics"]["games_played"] >= 3]
                
                if win_rates:
                    avg_win_rate = sum(win_rates) / len(win_rates)
                    best_win_rate = max(win_rates)
                    best_wilson = max(wilson_scores) if wilson_scores else 0
                    f.write(f"| {gen} | {len(strats)} | {avg_win_rate:.2f} | {best_win_rate:.2f} | {best_wilson:.4f} |\n")
                else:
                    f.write(f"| {gen} | {len(strats)} | N/A | N/A | N/A |\n")
            f.write("\n")
            
            # Include links to visualizations
            f.write("## Visualizations\n\n")
            vis_dir = os.path.join(self.output_dir, "visualizations")
            for img_file in os.listdir(vis_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join("visualizations", img_file)
                    f.write(f"![{img_file}]({img_path})\n\n")
        
        print(f"Training report generated at: {report_path}")
        return report_path
    
    def run_training(self):
        """
        Run the complete training process.
        """
        self.training_start_time = time.time()
        
        print(f"=== Starting Training (Generation {self.current_generation + 1}) ===")
        
        try:
            # Run for specified number of generations
            target_generations = self.current_generation + self.config["generations"]
            
            while self.current_generation < target_generations:
                self.current_generation += 1
                
                print(f"\n=== Generation {self.current_generation}/{target_generations} ===")
                
                # Generate team files for all strategies
                team_files = self.generate_team_files()
                
                # Run battles for this generation
                self.run_generation_battles(team_files)
                
                # Find generation champion
                self.find_generation_champion()
                
                # Evolve new strategies
                new_strategies = self.evolve_strategies()
                
                # Prune strategies to stay within max_strategies limit
                self.prune_strategies(new_strategies)
                
                # Save training state
                self.save_state()
                
                # Generate interim reports every 5 generations
                if self.current_generation % 5 == 0 or self.current_generation == target_generations:
                    self.analyzer.generate_reports()
            
            # Run final championship tournament
            print("\n=== Final Championship Tournament ===")
            final_champion = self.run_championship(top_n=5, battles_per_matchup=3)
            if final_champion:
                self.best_strategy_id = final_champion
            
            # Generate final submission
            self.generate_final_submission()
            
            # Generate comprehensive training report
            self.generate_training_report()
            
            print("\n=== Training Complete ===")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            # Save current state before exiting
            self.save_state()
            self.trainer.save_strategies()
            print("Training state and strategies saved.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tower defense strategy training')
    
    parser.add_argument('--output-dir', type=str, default='training_data',
                      help='Directory to store training data and results')
    parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode (auto-close windows)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create training manager with headless mode
    manager = TrainingManager(
        output_dir=args.output_dir,
        headless=args.headless
    )
    
    # Save the configuration (loaded from file)
    manager.save_config()
    
    # Run training
    manager.run_training()