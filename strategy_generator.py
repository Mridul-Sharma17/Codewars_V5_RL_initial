"""
Strategy generator for tower defense reinforcement learning system.
Creates new strategies through genetic algorithm techniques and mutation.
"""

import random
import json
import uuid
import copy
import math
from typing import Dict, List, Any, Tuple, Optional
from adaptive_templates import AdaptiveTemplates

class StrategyGenerator:
    """
    Generates new tower defense strategies through genetic algorithms and mutation.
    Handles strategy evolution, crossover, and mutation to discover optimal play strategies.
    """
    
    def __init__(self, template_type="adaptive", mutation_rate=0.25):
        """
        Initialize the strategy generator.
        
        Args:
            template_type: Default strategy template type
            mutation_rate: Probability of mutation for each parameter
        """
        self.template_type = template_type
        self.mutation_rate = mutation_rate
        self.strategy_counter = 0
        
        # Parameter bounds for mutation
        self.parameter_bounds = {
            "elixir_thresholds": {
                "min_deploy": (3.0, 6.0),
                "save_threshold": (5.0, 9.0),
                "emergency_threshold": (1.0, 3.0)
            },
            "position_settings": {
                "x_range": [(-30, -5), (5, 30)],  # Bounds for min and max values
                "y_default": (30, 48),
                "defensive_y": (15, 30)
            },
            "defensive_trigger": (0.3, 0.9),
            "counter_weights": {
                "air_vs_ground": (0.8, 2.0),
                "splash_vs_group": (1.0, 2.5),
                "tank_priority": (0.8, 1.8)
            },
            "timing_parameters": {
                "deployment_interval": (0.8, 2.5),
                "burst_threshold": (6, 10),
                "patience_factor": (0.4, 1.0)
            },
            "phase_parameters": {
                "early_game_threshold": (300, 700),
                "late_game_threshold": (900, 1500),
                "early_game_aggression": (0.1, 0.5),
                "mid_game_aggression": (0.4, 0.8),
                "late_game_aggression": (0.6, 1.0)
            },
            "memory_parameters": {
                "memory_length": (3, 8),
                "adaptation_rate": (0.3, 0.8)
            }
        }
        
        # Valid options for categorical parameters
        self.categorical_options = {
            "lane_preference": ["left", "right", "center", "split", "adaptive"],
            "template_type": ["counter_picker", "lane_adaptor", "elixir_optimizer", 
                             "phase_shifter", "pattern_recognition", "troop_synergy",
                             "adaptive", "baseline"]
        }
        
        # Troop ordering options
        self.available_troops = [
            "dragon", "wizard", "valkyrie", "musketeer",
            "knight", "archer", "minion", "barbarian"
        ]
    
    def generate_initial_population(self, size: int, base_strategies: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an initial population of strategies.
        
        Args:
            size: Number of strategies to generate
            base_strategies: Existing strategies to use as a base
            
        Returns:
            Dictionary of generated strategies
        """
        strategies = {}
        
        # Start with provided base strategies if available
        if base_strategies:
            strategies = copy.deepcopy(base_strategies)
            
        # Generate additional random strategies to reach desired size
        additional_needed = max(0, size - len(strategies))
        
        for _ in range(additional_needed):
            strategy_id, strategy = self.generate_random_strategy()
            strategies[strategy_id] = strategy
            
        return strategies
    
    def generate_random_strategy(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a completely random strategy.
        
        Returns:
            strategy_id, strategy_dict
        """
        self.strategy_counter += 1
        strategy_id = f"gen_{self._generate_id()}"
        
        # Create random parameter values
        lane_preference = random.choice(self.categorical_options["lane_preference"])
        template_type = random.choice(self.categorical_options["template_type"])
        
        # Generate random parameters within bounds
        elixir_thresholds = {
            "min_deploy": random.uniform(*self.parameter_bounds["elixir_thresholds"]["min_deploy"]),
            "save_threshold": random.uniform(*self.parameter_bounds["elixir_thresholds"]["save_threshold"]),
            "emergency_threshold": random.uniform(*self.parameter_bounds["elixir_thresholds"]["emergency_threshold"])
        }
        
        # Ensure logical relationship between thresholds
        elixir_thresholds["save_threshold"] = max(
            elixir_thresholds["save_threshold"], 
            elixir_thresholds["min_deploy"] + 1
        )
        
        position_settings = {
            "x_range": [
                random.randint(*self.parameter_bounds["position_settings"]["x_range"][0]),
                random.randint(*self.parameter_bounds["position_settings"]["x_range"][1])
            ],
            "y_default": random.randint(*self.parameter_bounds["position_settings"]["y_default"]),
            "defensive_y": random.randint(*self.parameter_bounds["position_settings"]["defensive_y"])
        }
        
        defensive_trigger = random.uniform(*self.parameter_bounds["defensive_trigger"])
        
        counter_weights = {
            "air_vs_ground": random.uniform(*self.parameter_bounds["counter_weights"]["air_vs_ground"]),
            "splash_vs_group": random.uniform(*self.parameter_bounds["counter_weights"]["splash_vs_group"]),
            "tank_priority": random.uniform(*self.parameter_bounds["counter_weights"]["tank_priority"])
        }
        
        # Generate random troop priority
        troop_priority = self.available_troops.copy()
        random.shuffle(troop_priority)
        
        # Random timing parameters
        timing_parameters = {
            "deployment_interval": random.uniform(*self.parameter_bounds["timing_parameters"]["deployment_interval"]),
            "burst_threshold": random.uniform(*self.parameter_bounds["timing_parameters"]["burst_threshold"]),
            "patience_factor": random.uniform(*self.parameter_bounds["timing_parameters"]["patience_factor"])
        }
        
        # Random phase parameters
        phase_parameters = {
            "early_game_threshold": random.uniform(*self.parameter_bounds["phase_parameters"]["early_game_threshold"]),
            "late_game_threshold": random.uniform(*self.parameter_bounds["phase_parameters"]["late_game_threshold"]),
            "early_game_aggression": random.uniform(*self.parameter_bounds["phase_parameters"]["early_game_aggression"]),
            "mid_game_aggression": random.uniform(*self.parameter_bounds["phase_parameters"]["mid_game_aggression"]),
            "late_game_aggression": random.uniform(*self.parameter_bounds["phase_parameters"]["late_game_aggression"])
        }
        
        # Ensure late game threshold is greater than early game threshold
        phase_parameters["late_game_threshold"] = max(
            phase_parameters["late_game_threshold"],
            phase_parameters["early_game_threshold"] + 300
        )
        
        # Random memory parameters for pattern recognition
        memory_parameters = {
            "memory_length": random.randint(*self.parameter_bounds["memory_parameters"]["memory_length"]),
            "adaptation_rate": random.uniform(*self.parameter_bounds["memory_parameters"]["adaptation_rate"])
        }
        
        # Create the strategy object
        strategy = {
            "id": strategy_id,
            "name": f"gen_{self.strategy_counter}",
            "template_type": template_type,
            "troop_selection": self.available_troops.copy(),  # We're using all available troops
            "lane_preference": lane_preference,
            "elixir_thresholds": elixir_thresholds,
            "position_settings": position_settings,
            "defensive_trigger": defensive_trigger,
            "counter_weights": counter_weights,
            "timing_parameters": timing_parameters,
            "phase_parameters": phase_parameters,
            "memory_parameters": memory_parameters,
            "troop_priority": troop_priority,
            "metrics": {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "avg_tower_health": 0,
                "win_rate": 0
            }
        }
        
        return strategy_id, strategy
    
    def evolve_population(self, 
                         strategies: Dict[str, Any], 
                         battle_metrics: List[Dict[str, Any]], 
                         generation: int,
                         population_size: int, 
                         elite_count: int = 2,
                         tournament_size: int = 3,
                         min_games_for_evolution: int = 3) -> Dict[str, Any]:
        """
        Evolve the population based on battle results.
        
        Args:
            strategies: Current generation strategies
            battle_metrics: Results of battles
            generation: Current generation number
            population_size: Desired size of the evolved population
            elite_count: Number of best strategies to preserve unchanged
            tournament_size: Number of strategies to include in each tournament selection
            min_games_for_evolution: Minimum number of games a strategy needs to have played
            
        Returns:
            Dictionary of evolved strategies
        """
        # Calculate fitness for each strategy
        fitness_scores = self._calculate_fitness(strategies, battle_metrics, min_games_for_evolution)
        
        # Sort strategies by fitness
        sorted_strategies = sorted(
            fitness_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select elites (best performing strategies)
        elites = []
        for i in range(min(elite_count, len(sorted_strategies))):
            if i < len(sorted_strategies):
                elite_id = sorted_strategies[i][0]
                elites.append(elite_id)
        
        # Create new population, starting with elites
        new_population = {}
        for elite_id in elites:
            if elite_id in strategies:
                new_population[elite_id] = copy.deepcopy(strategies[elite_id])
        
        # Fill the rest of the population through tournament selection and crossover
        while len(new_population) < population_size:
            # Select parents through tournament selection
            parent1_id = self._tournament_selection(fitness_scores, tournament_size)
            parent2_id = self._tournament_selection(fitness_scores, tournament_size)
            
            # Make sure parents are different
            attempts = 0
            while parent1_id == parent2_id and attempts < 5:
                parent2_id = self._tournament_selection(fitness_scores, tournament_size)
                attempts += 1
            
            # If we couldn't find different parents, generate a random strategy
            if parent1_id == parent2_id or parent1_id not in strategies or parent2_id not in strategies:
                child_id, child = self.generate_random_strategy()
                new_population[child_id] = child
                continue
            
            # Create child through crossover and mutation
            parent1 = strategies[parent1_id]
            parent2 = strategies[parent2_id]
            
            child_id, child = self._crossover_and_mutate(parent1, parent2, generation)
            new_population[child_id] = child
        
        # Add some completely random strategies to maintain diversity
        random_count = max(1, int(population_size * 0.1))  # 10% random
        for _ in range(random_count):
            if len(new_population) < population_size * 1.2:  # Allow up to 20% more strategies
                strategy_id, strategy = self.generate_random_strategy()
                new_population[strategy_id] = strategy
        
        return new_population
    
    def _calculate_fitness(self, 
                          strategies: Dict[str, Any], 
                          battle_metrics: List[Dict[str, Any]],
                          min_games: int = 3) -> Dict[str, float]:
        """
        Calculate fitness scores for all strategies based on battle results.
        
        Args:
            strategies: Current strategies
            battle_metrics: Battle results
            min_games: Minimum number of games needed for inclusion
            
        Returns:
            Dictionary mapping strategy IDs to fitness scores
        """
        # Count wins and games for each strategy
        wins = {}
        games_played = {}
        tower_health = {}
        
        for battle in battle_metrics:
            winner_id = battle.get("winner")
            loser_id = battle.get("loser")
            
            winner_health = battle.get("winner_tower_health", 10000)
            loser_health = battle.get("loser_tower_health", 0)
            
            # Track winner stats
            if winner_id:
                wins[winner_id] = wins.get(winner_id, 0) + 1
                games_played[winner_id] = games_played.get(winner_id, 0) + 1
                # Track average tower health (weighted average)
                if winner_id in tower_health:
                    prev_health, prev_count = tower_health[winner_id]
                    tower_health[winner_id] = (
                        (prev_health * prev_count + winner_health) / (prev_count + 1),
                        prev_count + 1
                    )
                else:
                    tower_health[winner_id] = (winner_health, 1)
            
            # Track loser stats
            if loser_id:
                games_played[loser_id] = games_played.get(loser_id, 0) + 1
                # Track average tower health (weighted average)
                if loser_id in tower_health:
                    prev_health, prev_count = tower_health[loser_id]
                    tower_health[loser_id] = (
                        (prev_health * prev_count + loser_health) / (prev_count + 1),
                        prev_count + 1
                    )
                else:
                    tower_health[loser_id] = (loser_health, 1)
        
        # Calculate fitness scores and update metrics
        fitness_scores = {}
        
        for strategy_id, strategy in strategies.items():
            # Get battle statistics
            strategy_wins = wins.get(strategy_id, 0)
            strategy_games = games_played.get(strategy_id, 0)
            
            # Update strategy metrics if it participated in battles
            if strategy_games > 0:
                # Update metrics in the strategy
                strategy["metrics"]["games_played"] = strategy_games + strategy["metrics"].get("games_played", 0)
                strategy["metrics"]["wins"] = strategy_wins + strategy["metrics"].get("wins", 0)
                strategy["metrics"]["losses"] = (strategy_games - strategy_wins) + strategy["metrics"].get("losses", 0)
                
                # Calculate win rate
                win_rate = strategy_wins / strategy_games if strategy_games > 0 else 0
                strategy["metrics"]["win_rate"] = win_rate
                
                # Get average tower health
                avg_health = tower_health.get(strategy_id, (0, 0))[0]
                strategy["metrics"]["avg_tower_health"] = avg_health
                
                # Only include strategies with enough games
                if strategy_games >= min_games:
                    # Fitness is primarily win rate, with a small bonus for tower health
                    health_factor = avg_health / 10000  # Normalize to 0-1
                    fitness = win_rate * 0.9 + health_factor * 0.1
                    
                    fitness_scores[strategy_id] = fitness
            else:
                # If strategy hasn't played in this generation, use historical data
                if strategy["metrics"]["games_played"] >= min_games:
                    win_rate = strategy["metrics"].get("win_rate", 0)
                    health_factor = strategy["metrics"].get("avg_tower_health", 0) / 10000
                    
                    # Apply a slight penalty for not being tested in current generation
                    fitness = (win_rate * 0.9 + health_factor * 0.1) * 0.95
                    fitness_scores[strategy_id] = fitness
                else:
                    # If strategy hasn't played enough games total, give it a neutral score
                    fitness_scores[strategy_id] = 0.5
        
        return fitness_scores
    
    def _tournament_selection(self, 
                             fitness_scores: Dict[str, float], 
                             tournament_size: int) -> str:
        """
        Select a strategy using tournament selection.
        
        Args:
            fitness_scores: Mapping of strategy IDs to fitness scores
            tournament_size: Number of strategies to include in the tournament
            
        Returns:
            Selected strategy ID
        """
        if not fitness_scores:
            return None
        
        # Select random candidates for the tournament
        candidates = list(fitness_scores.keys())
        if len(candidates) <= tournament_size:
            tournament = candidates
        else:
            tournament = random.sample(candidates, tournament_size)
        
        # Find the best strategy in the tournament
        best_strategy = None
        best_fitness = -1
        
        for strategy_id in tournament:
            fitness = fitness_scores.get(strategy_id, 0)
            if fitness > best_fitness:
                best_fitness = fitness
                best_strategy = strategy_id
        
        return best_strategy
    
    def _crossover_and_mutate(self, 
                             parent1: Dict[str, Any], 
                             parent2: Dict[str, Any],
                             generation: int) -> Tuple[str, Dict[str, Any]]:
        """
        Create a new strategy by crossing over two parent strategies and applying mutation.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            generation: Current generation number
            
        Returns:
            strategy_id, new_strategy
        """
        self.strategy_counter += 1
        child_id = f"gen_{self._generate_id()}"
        
        # Create child as a copy of parent1 initially
        child = copy.deepcopy(parent1)
        child["id"] = child_id
        child["name"] = f"gen_{generation}_{self.strategy_counter}"
        
        # Initialize new metrics
        child["metrics"] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "avg_tower_health": 0,
            "win_rate": 0
        }
        
        # Crossover - randomly select properties from each parent
        self._crossover_numeric_param(child, parent2, "defensive_trigger")
        
        # Crossover nested parameters
        for category in ["elixir_thresholds", "position_settings", "counter_weights", 
                         "timing_parameters", "phase_parameters", "memory_parameters"]:
            if category in parent1 and category in parent2:
                if category not in child:
                    child[category] = {}
                    
                for param in parent1[category]:
                    if param in parent2[category]:
                        # 50% chance to inherit from each parent
                        if random.random() < 0.5:
                            child[category][param] = copy.deepcopy(parent2[category][param])
        
        # Crossover categorical parameters
        for param in ["lane_preference", "template_type"]:
            if param in parent1 and param in parent2:
                if random.random() < 0.5:
                    child[param] = parent2[param]
        
        # Crossover troop priority
        if "troop_priority" in parent1 and "troop_priority" in parent2:
            # Uniform crossover of troop priorities
            child["troop_priority"] = []
            
            # Determine length (minimum of both parents)
            p1_troops = parent1["troop_priority"]
            p2_troops = parent2["troop_priority"]
            max_length = min(len(p1_troops), len(p2_troops))
            
            for i in range(max_length):
                if random.random() < 0.5 and i < len(p1_troops):
                    troop = p1_troops[i]
                elif i < len(p2_troops):
                    troop = p2_troops[i]
                else:
                    continue
                    
                if troop not in child["troop_priority"]:
                    child["troop_priority"].append(troop)
            
            # Add any missing troops
            all_troops = set(self.available_troops)
            for troop in all_troops:
                if troop not in child["troop_priority"]:
                    child["troop_priority"].append(troop)
        
        # Apply mutations
        self._mutate_strategy(child)
        
        return child_id, child
    
    def _crossover_numeric_param(self, 
                                child: Dict[str, Any], 
                                parent2: Dict[str, Any], 
                                param: str):
        """Helper method for crossing over a numeric parameter."""
        if param in child and param in parent2:
            # 50% chance to inherit from parent2
            if random.random() < 0.5:
                child[param] = parent2[param]
    
    def _mutate_strategy(self, strategy: Dict[str, Any]):
        """
        Apply random mutations to a strategy.
        
        Args:
            strategy: Strategy to mutate
        """
        # Mutate categorical parameters
        for param, options in self.categorical_options.items():
            if param in strategy and random.random() < self.mutation_rate:
                strategy[param] = random.choice(options)
        
        # Mutate numeric parameters
        if "defensive_trigger" in strategy and random.random() < self.mutation_rate:
            bounds = self.parameter_bounds["defensive_trigger"]
            strategy["defensive_trigger"] = self._mutate_numeric_value(
                strategy["defensive_trigger"], bounds[0], bounds[1]
            )
        
        # Mutate nested parameters
        self._mutate_nested_params(strategy, "elixir_thresholds", self.parameter_bounds["elixir_thresholds"])
        self._mutate_nested_params(strategy, "counter_weights", self.parameter_bounds["counter_weights"])
        self._mutate_nested_params(strategy, "timing_parameters", self.parameter_bounds["timing_parameters"])
        self._mutate_nested_params(strategy, "phase_parameters", self.parameter_bounds["phase_parameters"])
        self._mutate_nested_params(strategy, "memory_parameters", self.parameter_bounds["memory_parameters"])
        
        # Special handling for position settings
        if "position_settings" in strategy:
            # Mutate x_range specially because it's a list
            if "x_range" in strategy["position_settings"] and random.random() < self.mutation_rate:
                bounds_min = self.parameter_bounds["position_settings"]["x_range"][0]
                bounds_max = self.parameter_bounds["position_settings"]["x_range"][1]
                
                # Mutate min value
                strategy["position_settings"]["x_range"][0] = int(self._mutate_numeric_value(
                    strategy["position_settings"]["x_range"][0], bounds_min[0], bounds_min[1]
                ))
                
                # Mutate max value
                strategy["position_settings"]["x_range"][1] = int(self._mutate_numeric_value(
                    strategy["position_settings"]["x_range"][1], bounds_max[0], bounds_max[1]
                ))
            
            # Mutate other position settings
            for param in ["y_default", "defensive_y"]:
                if param in strategy["position_settings"] and random.random() < self.mutation_rate:
                    bounds = self.parameter_bounds["position_settings"][param]
                    strategy["position_settings"][param] = int(self._mutate_numeric_value(
                        strategy["position_settings"][param], bounds[0], bounds[1]
                    ))
        
        # Mutate troop priority
        if "troop_priority" in strategy and random.random() < self.mutation_rate:
            # Apply random swaps in the priority list
            num_swaps = random.randint(1, 3)
            priority_list = strategy["troop_priority"]
            
            for _ in range(num_swaps):
                if len(priority_list) >= 2:
                    i = random.randint(0, len(priority_list) - 1)
                    j = random.randint(0, len(priority_list) - 1)
                    priority_list[i], priority_list[j] = priority_list[j], priority_list[i]
    
    def _mutate_nested_params(self, strategy: Dict[str, Any], category: str, bounds: Dict[str, Any]):
        """Helper method for mutating nested parameters."""
        if category in strategy:
            for param, param_bounds in bounds.items():
                if param in strategy[category] and random.random() < self.mutation_rate:
                    min_val, max_val = param_bounds
                    
                    # Integer mutation for some specific parameters
                    if (category == "memory_parameters" and param == "memory_length" or
                        category == "phase_parameters" and param in ["early_game_threshold", "late_game_threshold"]):
                        strategy[category][param] = int(self._mutate_numeric_value(
                            strategy[category][param], min_val, max_val
                        ))
                    else:
                        strategy[category][param] = self._mutate_numeric_value(
                            strategy[category][param], min_val, max_val
                        )
    
    def _mutate_numeric_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Mutate a numeric value within bounds.
        
        Args:
            value: Current value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Mutated value
        """
        # Determine mutation magnitude based on parameter range
        range_size = max_val - min_val
        mutation_size = range_size * 0.2  # Allow mutation of up to 20% of the range
        
        # Apply random mutation
        delta = random.uniform(-mutation_size, mutation_size)
        new_value = value + delta
        
        # Keep within bounds
        new_value = max(min_val, min(max_val, new_value))
        
        return new_value
    
    def generate_strategy_code(self, strategy: Dict[str, Any]) -> str:
        """
        Generate Python code for a strategy using the appropriate template.
        
        Args:
            strategy: Strategy parameters
            
        Returns:
            Python code string for the strategy
        """
        # Default to adaptive template if none specified
        template_type = strategy.get("template_type", self.template_type)
        
        # Generate code using the appropriate template
        strategy_code = AdaptiveTemplates.generate_template_code(template_type, strategy)
        
        return strategy_code
    
    def _generate_id(self) -> str:
        """Generate a unique ID for a strategy."""
        # Generate random 8-character hex ID
        id_part = uuid.uuid4().hex[:8]
        return id_part