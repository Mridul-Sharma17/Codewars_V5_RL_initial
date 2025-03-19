"""
Advanced training system for tower defense strategy evolution.
Uses parallel processing, league system, and directed evolution to
create an unbeatable strategy.
"""

import os
import sys
import json
import time
import random
import shutil
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Import our custom modules
from game_runner import GameRunner
from improved_strategy_selection import StrategyEvaluator
from evolve import StrategyEvolver
from strategy_encoder import StrategyEncoder

class StrategyTrainer:
    """
    Main trainer class that coordinates the evolutionary process
    and tournament management.
    """
    
    def __init__(self, game_dir="game", strategies_file="training_data/strategies.json", 
                 config_file="training_data/training_config.json",
                 results_dir="training_data/results"):
        """
        Initialize the strategy trainer.
        
        Args:
            game_dir: Directory containing the game
            strategies_file: File to load/save strategies
            config_file: Training configuration file
            results_dir: Directory to save training results
        """
        self.game_dir = game_dir
        self.strategies_file = strategies_file
        self.config_file = config_file
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(strategies_file), exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.game_runner = GameRunner(game_path=game_dir, headless=True, max_workers=8)
        self.strategy_evaluator = StrategyEvaluator(confidence=0.95, elite_opponent_bonus=1.5)
        self.strategy_evolver = StrategyEvolver(mutation_rate=0.25, crossover_rate=0.75, 
                                               tournament_size=4, elite_preservation=0.15)
        self.encoder = StrategyEncoder(max_length=200)
        
        # Load training configuration
        self.config = self._load_config()
        
        # Track current generation
        self.current_generation = 0
        
        # Track battle results
        self.all_battle_results = []
        self.recent_battle_results = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from file"""
        default_config = {
            "generations": 10,
            "battles_per_generation": 20,
            "strategies_per_generation": 4,
            "min_games_for_evolution": 5,
            "tournament_size": 3
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                # Save default config
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            print(f"Error loading config: {e}")
        
        return default_config
    
    def load_strategies(self) -> Dict[str, Dict]:
        """
        Load strategies from the JSON file.
        Returns a dictionary of strategy_id -> strategy.
        """
        strategies = {}
        
        try:
            if os.path.exists(self.strategies_file):
                with open(self.strategies_file, 'r') as f:
                    strategies = json.load(f)
        except Exception as e:
            print(f"Error loading strategies: {e}")
        
        # If no strategies, create default seed strategies
        if not strategies:
            strategies = self._create_seed_strategies()
            self.save_strategies(strategies)
            
        return strategies
    
    def save_strategies(self, strategies: Dict[str, Dict]) -> None:
        """Save strategies to the JSON file"""
        try:
            with open(self.strategies_file, 'w') as f:
                json.dump(strategies, f, indent=2)
                print(f"Saved {len(strategies)} strategies to {self.strategies_file}")
        except Exception as e:
            print(f"Error saving strategies: {e}")
    
    def _create_seed_strategies(self) -> Dict[str, Dict]:
        """Create seed strategies for initial population"""
        strategies = {}
        
        # Base template for strategies
        template = {
            "name": "init",
            "lane_preference": "right",
            "elixir_thresholds": {
                "min_deploy": 4.0,
                "save_threshold": 7.0,
                "emergency_threshold": 2.0
            },
            "position_settings": {
                "x_range": [-20, 20],
                "y_default": 40,
                "defensive_y": 20
            },
            "defensive_trigger": 0.65,
            "counter_weights": {
                "air_vs_ground": 1.5,
                "splash_vs_group": 1.7,
                "tank_priority": 1.2
            },
            "troop_selection": [
                "dragon", "wizard", "valkyrie", "musketeer",
                "knight", "archer", "minion", "barbarian"
            ],
            "metrics": {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0
            }
        }
        
        # Create 5 diverse initial strategies
        # 1. Aggressive strategy
        aggressive = dict(template)
        aggressive["id"] = "aggressive"
        aggressive["name"] = "Aggressive"
        aggressive["lane_preference"] = "right"
        aggressive["elixir_thresholds"]["min_deploy"] = 3.0
        aggressive["position_settings"]["y_default"] = 45
        aggressive["defensive_trigger"] = 0.5
        aggressive["troop_priority"] = ["dragon", "wizard", "valkyrie", "musketeer", 
                                        "knight", "archer", "minion", "barbarian"]
        
        # 2. Defensive strategy
        defensive = dict(template)
        defensive["id"] = "defensive"
        defensive["name"] = "Defensive"
        defensive["lane_preference"] = "center"
        defensive["elixir_thresholds"]["min_deploy"] = 5.0
        defensive["position_settings"]["y_default"] = 35
        defensive["defensive_trigger"] = 0.8
        defensive["troop_priority"] = ["wizard", "musketeer", "knight", "valkyrie", 
                                      "dragon", "archer", "barbarian", "minion"]
        
        # 3. Balanced strategy
        balanced = dict(template)
        balanced["id"] = "balanced"
        balanced["name"] = "Balanced"
        balanced["lane_preference"] = "split"
        balanced["elixir_thresholds"]["min_deploy"] = 4.0
        balanced["position_settings"]["y_default"] = 40
        balanced["defensive_trigger"] = 0.65
        balanced["troop_priority"] = ["wizard", "dragon", "knight", "archer", 
                                     "valkyrie", "musketeer", "minion", "barbarian"]
        
        # 4. Anti-air strategy
        anti_air = dict(template)
        anti_air["id"] = "anti_air"
        anti_air["name"] = "AntiAir"
        anti_air["lane_preference"] = "right"
        anti_air["elixir_thresholds"]["min_deploy"] = 3.5
        anti_air["position_settings"]["y_default"] = 38
        anti_air["defensive_trigger"] = 0.6
        anti_air["counter_weights"]["air_vs_ground"] = 2.0
        anti_air["troop_priority"] = ["wizard", "musketeer", "archer", "dragon", 
                                     "knight", "valkyrie", "minion", "barbarian"]
        
        # 5. Tank-heavy strategy
        tank_heavy = dict(template)
        tank_heavy["id"] = "tank_heavy"
        tank_heavy["name"] = "TankHeavy"
        tank_heavy["lane_preference"] = "left"
        tank_heavy["elixir_thresholds"]["min_deploy"] = 4.5
        tank_heavy["position_settings"]["y_default"] = 42
        tank_heavy["defensive_trigger"] = 0.7
        tank_heavy["counter_weights"]["tank_priority"] = 2.0
        tank_heavy["troop_priority"] = ["knight", "valkyrie", "dragon", "wizard", 
                                       "musketeer", "barbarian", "archer", "minion"]
        
        strategies = {
            "aggressive": aggressive,
            "defensive": defensive,
            "balanced": balanced,
            "anti_air": anti_air,
            "tank_heavy": tank_heavy
        }
        
        # Add initial benchmark strategy (similar to Pratyaksh)
        benchmark = dict(template)
        benchmark["id"] = "benchmark"
        benchmark["name"] = "Benchmark"
        benchmark["lane_preference"] = "adaptive"
        benchmark["elixir_thresholds"]["min_deploy"] = 3.5
        benchmark["position_settings"]["y_default"] = 42
        benchmark["defensive_trigger"] = 0.6
        benchmark["counter_weights"]["air_vs_ground"] = 1.8
        benchmark["counter_weights"]["splash_vs_group"] = 2.0
        benchmark["troop_priority"] = ["wizard", "dragon", "musketeer", "valkyrie", 
                                       "knight", "archer", "minion", "barbarian"]
        strategies["benchmark"] = benchmark
        
        print(f"Created {len(strategies)} seed strategies")
        return strategies
    
    def _generate_strategy_scripts(self, strategies: Dict[str, Dict]) -> Dict[str, str]:
        """
        Generate Python scripts from strategy dictionaries.
        Returns a dictionary of strategy_id -> script_path.
        """
        script_paths = {}
        
        # Create temp directory for scripts
        script_dir = os.path.join("training_data", "scripts")
        os.makedirs(script_dir, exist_ok=True)
        
        for strategy_id, strategy in strategies.items():
            # Skip strategies without troop_priority
            if "troop_priority" not in strategy:
                print(f"Warning: Strategy {strategy_id} is missing troop_priority, skipping")
                continue
                
            # Generate team_signal parameter from strategy
            strategy["team_signal"] = self.encoder.encode(strategy)
            
            # Create script file path
            script_path = os.path.join(script_dir, f"{strategy_id}.py")
            
            # Generate Python code from strategy
            code = self._generate_strategy_code(strategy)
            
            # Write the code to file
            with open(script_path, 'w') as f:
                f.write(code)
            
            script_paths[strategy_id] = script_path
        
        # Always include Pratyaksh if available
        pratyaksh_path = self._find_pratyaksh_script()
        if pratyaksh_path:
            script_paths["pratyaksh"] = pratyaksh_path
        
        print(f"Generated {len(script_paths)} strategy scripts")
        return script_paths
    
    def _find_pratyaksh_script(self) -> str:
        """Find Pratyaksh script in various locations"""
        possible_paths = [
            "teams/pratyaksh.py",
            "pratyaksh.py",
            "teams/Pratyaksh.py",
            "Pratyaksh.py"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        return None
    
    def _generate_strategy_code(self, strategy: Dict[str, Any]) -> str:
        """Generate Python code for a strategy"""
        # Create a simplified troop selection array
        troop_selection = strategy.get("troop_selection", strategy.get("troop_priority", [
            "dragon", "wizard", "valkyrie", "musketeer",
            "knight", "archer", "minion", "barbarian"
        ]))
        
        # Ensure we have exactly 8 troops
        while len(troop_selection) < 8:
            missing_troops = [t for t in [
                "dragon", "wizard", "valkyrie", "musketeer",
                "knight", "archer", "minion", "barbarian"
            ] if t not in troop_selection]
            
            if not missing_troops:
                # Just duplicate the last troop
                troop_selection.append(troop_selection[-1])
            else:
                # Add a missing troop
                troop_selection.append(missing_troops[0])
                
        # Limit to exactly 8 troops
        troop_selection = troop_selection[:8]
        
        # Convert to proper case for imports
        proper_troops = [t.lower() for t in troop_selection]
        
        code = f'''from teams.helper_function import Troops, Utils
import random

team_name = "{strategy.get('name', 'Evolved')}"
troops = [
    Troops.{proper_troops[0]}, Troops.{proper_troops[1]}, 
    Troops.{proper_troops[2]}, Troops.{proper_troops[3]},
    Troops.{proper_troops[4]}, Troops.{proper_troops[5]}, 
    Troops.{proper_troops[6]}, Troops.{proper_troops[7]}
]
deploy_list = Troops([])
team_signal = "{strategy.get('team_signal', '')}"

# Strategy parameter constants
MIN_DEPLOY_ELIXIR = {strategy.get('elixir_thresholds', {}).get('min_deploy', 4.0)}
SAVE_THRESHOLD = {strategy.get('elixir_thresholds', {}).get('save_threshold', 7.0)}
DEFENSIVE_TRIGGER = {strategy.get('defensive_trigger', 0.65)}
Y_DEFAULT = {strategy.get('position_settings', {}).get('y_default', 40)}
Y_DEFENSIVE = {strategy.get('position_settings', {}).get('defensive_y', 20)}
X_MIN = {strategy.get('position_settings', {}).get('x_range', [-20, 20])[0]}
X_MAX = {strategy.get('position_settings', {}).get('x_range', [-20, 20])[1]}

def random_x(min_val=X_MIN, max_val=X_MAX):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    """
    DON'T TEMPER DEPLOY FUNCTION
    """
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Track opponent troops in team_signal
    for troop in opp_troops:
        if troop.name not in team_signal:
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += troop.name
    
    # Analyze whether to deploy troops based on elixir and game state
    if should_deploy(my_tower, opp_tower, opp_troops):
        # Choose best troop and position based on strategy
        troop, position = choose_troop_and_position(my_tower, opp_troops)
        if troop:
            deploy_troop(troop, position)

def should_deploy(my_tower, opp_tower, opp_troops):
    """Decide whether to deploy troops based on strategy and game state"""
    # Check if we have enough elixir
    if my_tower.total_elixir < MIN_DEPLOY_ELIXIR:
        return False
        
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
        
    # Always deploy early in the game
    if my_tower.game_timer < 300:
        return True
        
    # Deploy based on elixir availability
    if my_tower.total_elixir >= SAVE_THRESHOLD:
        return True
    elif my_tower.total_elixir >= MIN_DEPLOY_ELIXIR + 2:
        return random.random() > 0.3  # 70% chance to deploy
        
    # Default: deploy if we meet minimum elixir
    return my_tower.total_elixir >= MIN_DEPLOY_ELIXIR

def choose_troop_and_position(my_tower, opp_troops):
    """Choose the best troop to deploy and the position"""
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    my_tower_health_ratio = my_tower.health / 10000
    defensive_mode = my_tower_health_ratio < DEFENSIVE_TRIGGER
    
    # Choose troop based on strategy and available troops
    troop_priorities = {strategy.get('troop_priority', [])}
    
    # Score each available troop
    troop_scores = {{}}
    for troop in available_troops:
        # Higher score = higher priority
        if troop in troop_priorities:
            # Use the index in the priority list (reversed so earlier = higher score)
            troop_scores[troop] = len(troop_priorities) - troop_priorities.index(troop)
        else:
            # Default low score for troops not in priority list
            troop_scores[troop] = 0
    
    # Choose the highest scoring available troop
    chosen_troop = max(troop_scores.items(), key=lambda x: x[1])[0] if troop_scores else available_troops[0]
    
    # Determine position based on lane preference and mode
    lane_preference = "{strategy.get('lane_preference', 'right')}"
    x_pos = random_x()
    
    # Adjust position based on lane preference
    if lane_preference == "left":
        x_pos = random_x(X_MIN, -5)
    elif lane_preference == "right":
        x_pos = random_x(5, X_MAX)
    elif lane_preference == "center":
        x_pos = random_x(-10, 10)
    elif lane_preference == "split":
        # Alternate between left and right
        if random.random() > 0.5:
            x_pos = random_x(X_MIN, -5)
        else:
            x_pos = random_x(5, X_MAX)
    
    # Set y position based on defensive mode
    y_pos = Y_DEFENSIVE if defensive_mode else Y_DEFAULT
    
    return chosen_troop, (x_pos, y_pos)

def deploy_troop(troop, position):
    """Deploy the selected troop at the given position"""
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
'''
        return code

    def run_tournament(self, script_paths: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Run a round-robin tournament where each strategy plays against all others.
        
        Args:
            script_paths: Dictionary of strategy_id -> script_path
            
        Returns:
            List of battle result dictionaries
        """
        # Create all possible pairs without duplicates (round robin)
        battle_pairs = []
        strategy_ids = list(script_paths.keys())
        
        for i in range(len(strategy_ids)):
            for j in range(i + 1, len(strategy_ids)):
                # Add battle between strategies i and j
                battle_pairs.append((script_paths[strategy_ids[i]], script_paths[strategy_ids[j]]))
        
        print(f"Running {len(battle_pairs)} round-robin battles for generation {self.current_generation}")
        
        # Run the battles using parallel processing
        results_file = os.path.join(self.results_dir, f"gen_{self.current_generation}_results.json")
        battle_results = self.game_runner.run_parallel_battles(battle_pairs)
        
        # Save results
        if battle_results:
            with open(results_file, 'w') as f:
                json.dump(battle_results, f, indent=2)
        
        # Update our results tracking
        self.recent_battle_results = battle_results
        self.all_battle_results.extend(battle_results)
        
        return battle_results
    
    def _calculate_culling_rate(self, strategies: Dict[str, Dict]) -> float:
        """Calculate appropriate culling rate based on generation and population size"""
        # Base rate adjusted by generation (less aggressive early, more aggressive later)
        if self.current_generation <= 5:
            base_rate = 0.15  # Early generations: conservative culling (15%)
        elif self.current_generation <= 10:
            base_rate = 0.25  # Middle generations: moderate culling (25%)
        else:
            base_rate = 0.35  # Later generations: aggressive culling (35%)
        
        # Adjust based on population size
        population_size = len(strategies)
        if population_size > 20:
            # Very large population needs more pruning regardless of generation
            return min(0.4, base_rate + 0.1)
        elif population_size < 12:
            # Small population needs more conservative approach
            return max(0.1, base_rate - 0.1)
        
        return base_rate
    
    def _cull_poor_performers(self, strategies: Dict[str, Dict]) -> None:
        """
        Remove the bottom performers based on dynamic culling rate.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy
        """
        # Only keep strategies with enough games
        evaluated_strategies = {
            sid: strat for sid, strat in strategies.items()
            if strat.get("metrics", {}).get("games_played", 0) >= 3
        }
        
        # Don't cull if we have too few strategies
        min_population = 7  # Keep at least this many strategies
        if len(evaluated_strategies) <= min_population:
            return
        
        # Calculate culling rate
        culling_rate = self._calculate_culling_rate(strategies)
        
        # Sort by fitness score (using win rate and tower health)
        sorted_strategies = sorted(
            evaluated_strategies.items(),
            key=lambda x: (
                x[1].get("metrics", {}).get("win_rate", 0),
                x[1].get("metrics", {}).get("avg_tower_health", 0)
            ),
            reverse=False  # Sort ascending to find worst performers
        )
        
        # Calculate number to remove (based on calculated culling rate)
        remove_count = int(len(evaluated_strategies) * culling_rate)
        
        # Get IDs to remove
        to_remove = [sid for sid, _ in sorted_strategies[:remove_count]]
        
        # Don't remove initial seed strategies or current generation
        protected_ids = [
            "aggressive", "defensive", "balanced", "anti_air", 
            "tank_heavy", "benchmark", "pratyaksh", "pratyaksh_inspired",
            "rapid_cycle", "range_focused", "adaptive_position"
        ]
        
        final_remove = [sid for sid in to_remove 
                        if sid not in protected_ids and 
                        not f"gen_{self.current_generation}" in sid]
        
        # Remove from strategies dictionary
        for sid in final_remove:
            if sid in strategies:
                del strategies[sid]
        
        print(f"Culled {len(final_remove)} poor-performing strategies ({culling_rate:.1%} rate). Population size: {len(strategies)}")
    
    def update_strategy_metrics(self, strategies: Dict[str, Dict]) -> None:
        """
        Update strategy metrics based on battle results.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy
        """
        # First reset metrics for all strategies
        for strategy in strategies.values():
            strategy["metrics"] = {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "avg_tower_health": 0
            }
        
        # Track strategy file names to IDs
        file_to_id = {f"{strategy_id}.py": strategy_id for strategy_id in strategies}
        
        # Process all battle results
        total_battles = len(self.all_battle_results)
        processed = 0
        
        for battle in self.all_battle_results:
            team1_file = battle.get("team1_file", "")
            team2_file = battle.get("team2_file", "")
            
            # Extract strategy IDs
            team1_id = file_to_id.get(team1_file, None)
            team2_id = file_to_id.get(team2_file, None)
            
            if not team1_id and ".py" in team1_file:
                # Try matching by removing extension and path
                base_name = os.path.splitext(os.path.basename(team1_file))[0]
                if base_name in strategies:
                    team1_id = base_name
            
            if not team2_id and ".py" in team2_file:
                # Try matching by removing extension and path
                base_name = os.path.splitext(os.path.basename(team2_file))[0]
                if base_name in strategies:
                    team2_id = base_name
            
            if not team1_id or not team2_id:
                continue  # Skip if we can't identify both strategies
                
            # Update metrics for team 1
            if team1_id in strategies:
                strategy = strategies[team1_id]
                strategy["metrics"]["games_played"] += 1
                if battle.get("winner") == battle.get("team1_name"):
                    strategy["metrics"]["wins"] += 1
                else:
                    strategy["metrics"]["losses"] += 1
                    
                if "team1_health" in battle:
                    # Running average of tower health
                    current_avg = strategy["metrics"]["avg_tower_health"]
                    games = strategy["metrics"]["games_played"]
                    new_health = battle["team1_health"]
                    
                    if games > 1:
                        strategy["metrics"]["avg_tower_health"] = (
                            (current_avg * (games - 1) + new_health) / games
                        )
                    else:
                        strategy["metrics"]["avg_tower_health"] = new_health
            
            # Update metrics for team 2
            if team2_id in strategies:
                strategy = strategies[team2_id]
                strategy["metrics"]["games_played"] += 1
                if battle.get("winner") == battle.get("team2_name"):
                    strategy["metrics"]["wins"] += 1
                else:
                    strategy["metrics"]["losses"] += 1
                    
                if "team2_health" in battle:
                    # Running average of tower health
                    current_avg = strategy["metrics"]["avg_tower_health"]
                    games = strategy["metrics"]["games_played"]
                    new_health = battle["team2_health"]
                    
                    if games > 1:
                        strategy["metrics"]["avg_tower_health"] = (
                            (current_avg * (games - 1) + new_health) / games
                        )
                    else:
                        strategy["metrics"]["avg_tower_health"] = new_health
            
            processed += 1
        
        # Calculate win rates
        for strategy in strategies.values():
            games = strategy["metrics"]["games_played"]
            if games > 0:
                strategy["metrics"]["win_rate"] = strategy["metrics"]["wins"] / games
        
        print(f"Updated metrics for {len(strategies)} strategies based on {processed}/{total_battles} battles")
    
    def evolve_strategies(self, strategies: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Evolve strategies based on battle results.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy
            
        Returns:
            Updated dictionary with new strategies
        """
        # Get sorted strategies by fitness
        sorted_strategies = self.strategy_evaluator.select_best_strategies(self.all_battle_results)
        
        # Calculate league assignments for each strategy
        for strategy_id in strategies:
            strategies[strategy_id]["league"] = self.strategy_evaluator.get_league_assignment(
                strategy_id, self.all_battle_results, min_games=self.config["min_games_for_evolution"]
            )
        
        # Generate offspring
        num_offspring = self.config["strategies_per_generation"]
        
        # Create next generation
        next_gen = self.strategy_evolver.evolve_population(
            strategies, num_offspring=num_offspring,
            current_generation=self.current_generation + 1
        )
        
        if not next_gen:
            print("Warning: No offspring generated")
            return strategies
            
        # Add new strategies to our collection
        for new_strategy in next_gen:
            strategy_id = new_strategy["id"]
            strategies[strategy_id] = new_strategy
        
        print(f"Evolved {len(next_gen)} new strategies for generation {self.current_generation + 1}")
        return strategies
    
    def run_training(self, generations=None, continue_training=False) -> Dict[str, Dict]:
        """
        Run the full training process for multiple generations.
        
        Args:
            generations: Number of generations to run (None = use config)
            continue_training: Whether to continue from last generation
            
        Returns:
            Dictionary of final evolved strategies
        """
        # Load existing strategies
        strategies = self.load_strategies()
        
        if generations is None:
            generations = self.config["generations"]
            
        print(f"Starting training for {generations} generations")
        print(f"Initial population: {len(strategies)} strategies")
        
        # Determine starting generation
        start_gen = 0
        if continue_training:
            # Find the highest generation number in strategies
            for strategy in strategies.values():
                if "generation" in strategy:
                    start_gen = max(start_gen, strategy["generation"])
            self.current_generation = start_gen
            print(f"Continuing training from generation {start_gen}")
        
        # Main training loop
        for gen in range(start_gen, start_gen + generations):
            self.current_generation = gen
            print(f"\n===== Generation {gen} =====")
            
            # Generate scripts from strategies
            script_paths = self._generate_strategy_scripts(strategies)
            
            # Run tournament
            battle_results = self.run_tournament(script_paths)
            
            # Update strategy metrics
            self.update_strategy_metrics(strategies)
            
            # Print current standings
            self._print_current_standings(strategies)
            
            # Apply culling if we're past the first few generations
            if self.current_generation >= 3:
                self._cull_poor_performers(strategies)
            
            # Save current state
            self.save_strategies(strategies)
            
            # Evolve strategies for next generation
            strategies = self.evolve_strategies(strategies)
            
        print("\nTraining complete!")
        
        # Generate final best strategy script
        best_strategy = self._select_final_strategy(strategies)
        if best_strategy:
            print("\nFinal best strategy:")
            print(f"ID: {best_strategy['id']}")
            print(f"Name: {best_strategy['name']}")
            print(f"Games: {best_strategy['metrics']['games_played']}")
            print(f"Win rate: {best_strategy['metrics']['win_rate']:.2f}")
            
            # Save best strategy to final script
            self._save_final_strategy(best_strategy)
        
        return strategies
    
    def _print_current_standings(self, strategies: Dict[str, Dict]) -> None:
        """Print current strategy standings"""
        # Sort by win rate
        sorted_strategies = sorted(
            strategies.items(),
                        key=lambda x: x[1]["metrics"].get("win_rate", 0),
            reverse=True
        )
        
        print("\nCurrent Strategy Rankings:")
        print("-" * 70)
        print(f"{'Rank':<6}{'Strategy ID':<20}{'Games':<8}{'Win Rate':<10}{'Avg Health':<12}League")
        print("-" * 70)
        
        for i, (strategy_id, strategy) in enumerate(sorted_strategies):
            metrics = strategy["metrics"]
            games = metrics.get("games_played", 0)
            win_rate = metrics.get("win_rate", 0)
            health = metrics.get("avg_tower_health", 0)
            league = strategy.get("league", "unranked")
            
            # Skip strategies with no games
            if games == 0:
                continue
                
            print(f"{i+1:<6}{strategy_id:<20}{games:<8}{win_rate:.3f}{health:>10.1f}  {league}")
        
        print("-" * 70)
    
    def _select_final_strategy(self, strategies: Dict[str, Dict]) -> Dict[str, Any]:
        """Select the best strategy from all candidates"""
        min_games = 5  # Minimum games to be considered
        
        # Filter strategies with enough games
        candidates = {
            sid: strat for sid, strat in strategies.items()
            if strat["metrics"]["games_played"] >= min_games
        }
        
        if not candidates:
            print("No strategies with enough games to select final strategy")
            return None
        
        # Calculate extended fitness score
        for sid, strat in candidates.items():
            # Base score is win rate
            win_rate = strat["metrics"].get("win_rate", 0)
            games_played = strat["metrics"].get("games_played", 0)
            avg_health = strat["metrics"].get("avg_tower_health", 0) / 10000  # Normalize
            
            # Calculate confidence factor (more games = more confidence)
            # Scales from 0.8 to 1.0 as games increases
            confidence_factor = 0.8 + min(0.2, (games_played - min_games) * 0.01)
            
            # Calculate health bonus (0 to 0.1 based on avg health preservation)
            health_bonus = min(0.1, avg_health * 0.1)
            
            # Special bonus if it beat Pratyaksh consistently
            pratyaksh_bonus = 0
            if hasattr(self, "recent_battle_results"):
                # Check wins against Pratyaksh
                wins_vs_pratyaksh = 0
                games_vs_pratyaksh = 0
                
                for battle in self.all_battle_results:
                    # Check if this strategy played against Pratyaksh
                    is_team1 = battle.get("team1_file", "").endswith(f"{sid}.py")
                    is_team2 = battle.get("team2_file", "").endswith(f"{sid}.py")
                    
                    vs_pratyaksh = (
                        (is_team1 and "pratyaksh" in battle.get("team2_file", "").lower()) or
                        (is_team2 and "pratyaksh" in battle.get("team1_file", "").lower())
                    )
                    
                    if vs_pratyaksh:
                        games_vs_pratyaksh += 1
                        if ((is_team1 and battle.get("team1_win", 0) == 1) or
                            (is_team2 and battle.get("team2_win", 0) == 1)):
                            wins_vs_pratyaksh += 1
                
                # Calculate Pratyaksh bonus (up to 0.2 extra)
                if games_vs_pratyaksh >= 2:
                    pratyaksh_bonus = 0.2 * (wins_vs_pratyaksh / games_vs_pratyaksh)
            
            # Final fitness score
            strat["final_fitness"] = (
                win_rate * confidence_factor + 
                health_bonus + 
                pratyaksh_bonus
            )
        
        # Find best strategy by final fitness
        best_strategy = max(candidates.values(), key=lambda s: s.get("final_fitness", 0))
        
        return best_strategy
    
    def _save_final_strategy(self, strategy: Dict[str, Any]) -> None:
        """Save the best strategy as the final submission"""
        # Get optimized parameters from strategy
        min_deploy = strategy.get("elixir_thresholds", {}).get("min_deploy", 3.75)
        save_threshold = strategy.get("elixir_thresholds", {}).get("save_threshold", 7.5)
        emergency_threshold = strategy.get("elixir_thresholds", {}).get("emergency_threshold", 2.0)
        defensive_trigger = strategy.get("defensive_trigger", 0.65)
        x_min = strategy.get("position_settings", {}).get("x_range", [-20, 20])[0]
        x_max = strategy.get("position_settings", {}).get("x_range", [-20, 20])[1]
        y_default = strategy.get("position_settings", {}).get("y_default", 40)
        y_defensive = strategy.get("position_settings", {}).get("defensive_y", 20)
        y_aggressive = strategy.get("position_settings", {}).get("y_aggressive", 47)
        
        # Get counter weights
        air_vs_ground = strategy.get("counter_weights", {}).get("air_vs_ground", 1.5)
        splash_vs_group = strategy.get("counter_weights", {}).get("splash_vs_group", 2.0)
        tank_priority = strategy.get("counter_weights", {}).get("tank_priority", 1.3)
        
        # Get troop priority
        troop_priority = strategy.get("troop_priority", [
            "dragon", "wizard", "valkyrie", "musketeer",
            "knight", "archer", "minion", "barbarian"
        ])
        troop_priority_str = ", ".join([f'"{t}"' for t in troop_priority[:8]])
        
        # Create team signal with key parameters
        troop_priority_sig = ""
        for t in troop_priority[:8]:
            troop_priority_sig += t[0].lower() if t else "x"
        
        team_signal = f"v:1,l:a,e:{min_deploy:.2f},s:{save_threshold:.2f},dt:{defensive_trigger:.2f},"
        team_signal += f"y:{y_default},xl:{x_min},xr:{x_max},tp:{troop_priority_sig},"
        team_signal += f"ca:{air_vs_ground:.1f},cs:{splash_vs_group:.1f},ct:{tank_priority:.1f}"
        
        # Create final strategy code from template
        with open("adaptive_tower_defense.py", 'r') as template_file:
            template = template_file.read()
            
            # Replace parameters in the template
            final_code = template.replace('team_name = "UnstoppableForce"', f'team_name = "{strategy.get("name", "OptimizedStrategy")}"')
            final_code = final_code.replace('team_signal = "v:1,l:a,e:3.75,s:7.5,dt:0.65,y:40,xl:-20,xr:20,tp:dwvmkab,ca:1.5,cs:2.0,ct:1.3"', f'team_signal = "{team_signal}"')
            
            # Update PARAMS values
            final_code = final_code.replace('"MIN_DEPLOY_ELIXIR": 3.75', f'"MIN_DEPLOY_ELIXIR": {min_deploy}')
            final_code = final_code.replace('"SAVE_THRESHOLD": 7.5', f'"SAVE_THRESHOLD": {save_threshold}')
            final_code = final_code.replace('"EMERGENCY_THRESHOLD": 2.0', f'"EMERGENCY_THRESHOLD": {emergency_threshold}')
            final_code = final_code.replace('"X_RANGE_LEFT": -20', f'"X_RANGE_LEFT": {x_min}')
            final_code = final_code.replace('"X_RANGE_RIGHT": 20', f'"X_RANGE_RIGHT": {x_max}')
            final_code = final_code.replace('"Y_DEFAULT": 40', f'"Y_DEFAULT": {y_default}')
            final_code = final_code.replace('"Y_DEFENSIVE": 20', f'"Y_DEFENSIVE": {y_defensive}')
            final_code = final_code.replace('"Y_AGGRESSIVE": 47', f'"Y_AGGRESSIVE": {y_aggressive}')
            final_code = final_code.replace('"DEFENSIVE_TRIGGER": 0.65', f'"DEFENSIVE_TRIGGER": {defensive_trigger}')
            final_code = final_code.replace('"AIR_VS_GROUND_BONUS": 1.5', f'"AIR_VS_GROUND_BONUS": {air_vs_ground}')
            final_code = final_code.replace('"SPLASH_VS_GROUP_BONUS": 2.0', f'"SPLASH_VS_GROUP_BONUS": {splash_vs_group}')
            final_code = final_code.replace('"TANK_PRIORITY_BONUS": 1.3', f'"TANK_PRIORITY_BONUS": {tank_priority}')
        
        # Save the optimized submission file
        submission_file = "optimized_submission.py"
        with open(submission_file, 'w') as f:
            f.write(final_code)
            
        print(f"Final optimized strategy saved to {submission_file}")

        # Also save a copy with the date and win rate for tracking
        win_rate = strategy["metrics"]["win_rate"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        backup_file = f"optimized_submission_{timestamp}_{win_rate:.3f}.py"
        shutil.copy(submission_file, backup_file)
        print(f"Backup saved as {backup_file}")
        
        
class StrategyEncoder:
    """
    Encodes strategy parameters into compact team_signal format and back.
    """
    
    def __init__(self, max_length=200):
        """Initialize the encoder with a maximum length constraint"""
        self.max_length = max_length
        
    def encode(self, strategy: Dict[str, Any]) -> str:
        """Encode strategy parameters as a compact string"""
        # Create encoded parts
        parts = []
        
        # Version
        parts.append("v:1")
        
        # Lane preference
        lane_pref = strategy.get("lane_preference", "adaptive")
        parts.append(f"l:{lane_pref[0]}")
        
        # Elixir thresholds
        min_deploy = strategy.get("elixir_thresholds", {}).get("min_deploy", 4.0)
        parts.append(f"e:{min_deploy:.2f}")
        
        save_threshold = strategy.get("elixir_thresholds", {}).get("save_threshold", 7.0)
        parts.append(f"s:{save_threshold:.2f}")
        
        # Defensive trigger
        defensive_trigger = strategy.get("defensive_trigger", 0.65)
        parts.append(f"dt:{defensive_trigger:.2f}")
        
        # Position settings
        y_default = strategy.get("position_settings", {}).get("y_default", 40) 
        parts.append(f"y:{y_default}")
        
        x_range = strategy.get("position_settings", {}).get("x_range", [-20, 20])
        parts.append(f"xl:{x_range[0]}")
        parts.append(f"xr:{x_range[1]}")
        
        # Troop priority as abbreviations
        troop_priority = strategy.get("troop_priority", [])
        if troop_priority:
            abbr = ""
            for troop in troop_priority[:8]:  # Limit to 8 troops
                if troop:
                    abbr += troop[0].lower()
                else:
                    abbr += "x"
            parts.append(f"tp:{abbr}")
        
        # Counter weights
        air_vs_ground = strategy.get("counter_weights", {}).get("air_vs_ground", 1.5)
        parts.append(f"ca:{air_vs_ground:.1f}")
        
        splash_vs_group = strategy.get("counter_weights", {}).get("splash_vs_group", 1.7)
        parts.append(f"cs:{splash_vs_group:.1f}")
        
        tank_priority = strategy.get("counter_weights", {}).get("tank_priority", 1.2)
        parts.append(f"ct:{tank_priority:.1f}")
        
        # Join and ensure within max length
        signal = ",".join(parts)
        if len(signal) > self.max_length:
            signal = signal[:self.max_length]
            
        return signal
        
    def decode(self, signal: str) -> Dict[str, Any]:
        """Decode a strategy from encoded team_signal string"""
        parts = signal.split(",")
        strategy = {
            "elixir_thresholds": {},
            "position_settings": {"x_range": [-20, 20]},
            "counter_weights": {}
        }
        
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            
            if key == "v":  # Version
                strategy["version"] = value
            elif key == "l":  # Lane preference
                if value == "l":
                    strategy["lane_preference"] = "left"
                elif value == "r":
                    strategy["lane_preference"] = "right"
                elif value == "c":
                    strategy["lane_preference"] = "center"
                elif value == "s":
                    strategy["lane_preference"] = "split"
                else:
                    strategy["lane_preference"] = "adaptive"
            elif key == "e":  # Min deploy elixir
                try:
                    strategy["elixir_thresholds"]["min_deploy"] = float(value)
                except ValueError:
                    pass
            elif key == "s":  # Save threshold
                try:
                    strategy["elixir_thresholds"]["save_threshold"] = float(value)
                except ValueError:
                    pass
            elif key == "dt":  # Defensive trigger
                try:
                    strategy["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # Default y position
                try:
                    strategy["position_settings"]["y_default"] = int(value)
                except ValueError:
                    pass
            elif key == "xl":  # X left bound
                try:
                    strategy["position_settings"]["x_range"][0] = int(value)
                except ValueError:
                    pass
            elif key == "xr":  # X right bound
                try:
                    strategy["position_settings"]["x_range"][1] = int(value)
                except ValueError:
                    pass
            elif key == "tp":  # Troop priority
                troop_mapping = {
                    "d": "dragon",
                    "w": "wizard",
                    "v": "valkyrie",
                    "m": "musketeer",
                    "k": "knight",
                    "a": "archer",
                    "i": "minion",
                    "b": "barbarian",
                    "p": "prince",
                    "g": "giant",
                    "s": "skeleton",
                    "l": "balloon"
                }
                
                strategy["troop_priority"] = []
                for char in value:
                    if char.lower() in troop_mapping:
                        strategy["troop_priority"].append(troop_mapping[char.lower()])
            elif key == "ca":  # Air vs ground bonus
                try:
                    strategy["counter_weights"]["air_vs_ground"] = float(value)
                except ValueError:
                    pass
            elif key == "cs":  # Splash vs group bonus
                try:
                    strategy["counter_weights"]["splash_vs_group"] = float(value)
                except ValueError:
                    pass
            elif key == "ct":  # Tank priority bonus
                try:
                    strategy["counter_weights"]["tank_priority"] = float(value)
                except ValueError:
                    pass
        
        return strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evolve tower defense strategies")
    parser.add_argument("--generations", type=int, help="Number of generations to train")
    parser.add_argument("--continue", action="store_true", help="Continue training from last generation")
    parser.add_argument("--game-dir", default="game", help="Game directory path")
    parser.add_argument("--strategies-file", default="training_data/strategies.json", 
                       help="File to save/load strategies")
    parser.add_argument("--config-file", default="training_data/training_config.json",
                       help="Training configuration file")
    parser.add_argument("--results-dir", default="training_data/results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = StrategyTrainer(
        game_dir=args.game_dir, 
        strategies_file=args.strategies_file,
        config_file=args.config_file,
        results_dir=args.results_dir
    )
    
    # Run training
    trainer.run_training(
        generations=args.generations,
        continue_training=getattr(args, "continue")
    )
    
    print("\nTraining complete! Final optimized strategy is ready for submission.")