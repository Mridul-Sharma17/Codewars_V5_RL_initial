import random
import copy
import uuid
import numpy as np
import math
from typing import List, Dict, Any, Tuple

class StrategyEvolver:
    """
    Implements genetic algorithm techniques to evolve tower defense strategies.
    Creates new strategies by selecting, crossing over, and mutating successful ones.
    """
    
    def __init__(self, mutation_rate=0.3, crossover_rate=0.7, tournament_size=3):
        """
        Initialize the strategy evolver.
        
        Args:
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover between parents
            tournament_size: Number of strategies to consider in tournament selection
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Parameter mutation ranges - limits how much each parameter can change
        self.param_ranges = {
            "elixir_thresholds.min_deploy": (1, 8, 0.5),     # (min, max, step)
            "elixir_thresholds.save_threshold": (5, 10, 0.5),
            "elixir_thresholds.emergency_threshold": (1, 5, 0.5),
            "position_settings.x_range.0": (-30, 0, 1),      # First value in x_range
            "position_settings.x_range.1": (0, 30, 1),       # Second value in x_range
            "position_settings.y_default": (10, 49, 2),
            "position_settings.defensive_y": (0, 30, 2),
            "defensive_trigger": (0.2, 0.9, 0.05),
            "counter_weights.air_vs_ground": (0.5, 3.0, 0.1),
            "counter_weights.splash_vs_group": (0.5, 3.0, 0.1),
            "counter_weights.tank_priority": (0.5, 3.0, 0.1)
        }
        
        # Categorical parameter options
        self.categorical_options = {
            "lane_preference": ["left", "right", "center", "split"]
        }
        
    def evolve_population(self, strategies: Dict[str, Dict], num_offspring: int = 4) -> List[Dict]:
        """
        Create a new generation of strategies through evolution.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy
            num_offspring: Number of new strategies to create
            
        Returns:
            List of new evolved strategies
        """
        # Convert strategies dict to list, filtering out those with no games
        strategy_list = [s for s in strategies.values() 
                        if s["metrics"]["games_played"] > 0]
        
        if len(strategy_list) < 2:
            print("Not enough strategies with games played for evolution")
            return []
        
        # Create offspring
        offspring = []
        for _ in range(num_offspring):
            # Select parents using tournament selection
            parent1 = self._tournament_selection(strategy_list)
            parent2 = self._tournament_selection(strategy_list)
            
            # Avoid selecting the same parent twice
            attempts = 0
            while parent2["id"] == parent1["id"] and attempts < 5:
                parent2 = self._tournament_selection(strategy_list)
                attempts += 1
            
            # Create child through crossover and mutation
            if random.random() < self.crossover_rate:
                # Crossover
                child = self._crossover(parent1, parent2)
            else:
                # Just use one parent as template
                child = copy.deepcopy(parent1)
            
            # Apply mutations
            child = self._mutate(child)
            
            # Reset metrics and generate new ID
            child["id"] = f"gen_{str(uuid.uuid4())[:8]}"
            child["name"] = f"evolved_{len(offspring) + 1}"
            child["metrics"] = {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "avg_tower_health": 0,
                "win_rate": 0
            }
            
            offspring.append(child)
            
        print(f"Created {len(offspring)} new evolved strategies")
        return offspring
    
    def _tournament_selection(self, strategies: List[Dict]) -> Dict:
        """
        Select a strategy using tournament selection with Wilson score.
        
        Args:
            strategies: List of strategies to select from
            
        Returns:
            Selected strategy
        """
        # Import Wilson score if available, otherwise use win rate
        try:
            from improved_strategy_selection import calculate_wilson_score
            use_wilson = True
        except ImportError:
            use_wilson = False
        
        # Adjust tournament size if we have fewer strategies
        actual_tournament_size = min(self.tournament_size, len(strategies))
        
        # Randomly select strategies for tournament
        tournament = random.sample(strategies, actual_tournament_size)
        
        # Find the best strategy in the tournament
        best_strategy = None
        best_fitness = -1
        
        for strategy in tournament:
            metrics = strategy["metrics"]
            wins = metrics.get("wins", 0)
            games_played = metrics.get("games_played", 0)
            
            if games_played > 0:
                # Calculate fitness using Wilson score if available
                if use_wilson and games_played >= 3:
                    fitness = calculate_wilson_score(wins, games_played)
                else:
                    # Fallback to win rate with a small bonus for experience
                    win_rate = wins / games_played
                    fitness = win_rate + (0.01 * min(games_played, 10))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = strategy
        
        # If no strategy with games played, pick randomly
        if best_strategy is None and tournament:
            best_strategy = random.choice(tournament)
            
        return best_strategy
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Create a new strategy by combining two parent strategies.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            New strategy created through crossover
        """
        # Create a deep copy of parent1 as the base
        child = copy.deepcopy(parent1)
        
        # 1. Crossover nested dictionary parameters
        for main_key in ["elixir_thresholds", "position_settings", "counter_weights"]:
            if main_key in parent1 and main_key in parent2:
                # 50% chance to take each sub-parameter from either parent
                for sub_key in parent1[main_key]:
                    if sub_key in parent2[main_key] and random.random() > 0.5:
                        # Special handling for x_range which is a list
                        if sub_key == "x_range" and isinstance(parent2[main_key][sub_key], list):
                            # 50% chance to take individual elements
                            for i in range(len(child[main_key][sub_key])):
                                if i < len(parent2[main_key][sub_key]) and random.random() > 0.5:
                                    child[main_key][sub_key][i] = parent2[main_key][sub_key][i]
                        else:
                            child[main_key][sub_key] = parent2[main_key][sub_key]
        
        # 2. Crossover other top-level parameters
        for key in ["lane_preference", "defensive_trigger"]:
            if key in parent1 and key in parent2 and random.random() > 0.5:
                child[key] = parent2[key]
        
        # 3. Special handling for troop_priority if present
        if "troop_priority" in parent1 and "troop_priority" in parent2:
            # Use crossover point in the list
            crossover_point = random.randint(1, min(len(parent1["troop_priority"]), 
                                                  len(parent2["troop_priority"])) - 1)
            child["troop_priority"] = (
                parent1["troop_priority"][:crossover_point] + 
                parent2["troop_priority"][crossover_point:]
            )
            
            # Ensure no duplicates
            seen = set()
            unique_troops = []
            for troop in child["troop_priority"]:
                if troop not in seen:
                    seen.add(troop)
                    unique_troops.append(troop)
            
            # If we lost troops due to deduplication, add missing ones from parent1
            if len(unique_troops) < len(parent1["troop_priority"]):
                for troop in parent1["troop_priority"]:
                    if troop not in seen and len(unique_troops) < len(parent1["troop_priority"]):
                        unique_troops.append(troop)
                        seen.add(troop)
            
            child["troop_priority"] = unique_troops
            
        return child
    
    def _mutate(self, strategy: Dict) -> Dict:
        """
        Apply mutations to a strategy.
        
        Args:
            strategy: Strategy to mutate
            
        Returns:
            Mutated strategy
        """
        # Create a deep copy to avoid modifying the original
        mutated = copy.deepcopy(strategy)
        
        # 1. Mutate numeric parameters with defined ranges
        for param_path, (min_val, max_val, step) in self.param_ranges.items():
            # Only mutate with probability = mutation_rate
            if random.random() > self.mutation_rate:
                continue
                
            # Parse the parameter path (e.g., "elixir_thresholds.min_deploy")
            path_parts = param_path.split(".")
            
            # Navigate to the right part of the dictionary
            target = mutated
            for part in path_parts[:-1]:
                if part.isdigit():  # Handle numeric indices for lists
                    part = int(part)
                if part not in target:
                    # Skip if the path doesn't exist
                    target = None
                    break
                target = target[part]
            
            if target is None:
                continue
                
            # Get the final key
            final_key = path_parts[-1]
            if final_key.isdigit():  # Handle numeric indices for lists
                final_key = int(final_key)
                
            if final_key not in target:
                continue
                
            # Current value
            current_val = target[final_key]
            
            # Apply mutation
            if isinstance(step, int):
                # Integer parameter
                new_val = current_val + random.randint(-2, 2) * step
                new_val = max(min_val, min(max_val, new_val))
                target[final_key] = int(new_val)
            else:
                # Float parameter
                new_val = current_val + (random.random() * 2 - 1) * step * 2  # +/- 2*step
                new_val = max(min_val, min(max_val, new_val))
                target[final_key] = new_val
        
        # 2. Mutate categorical parameters
        for param, options in self.categorical_options.items():
            if param in mutated and random.random() < self.mutation_rate:
                mutated[param] = random.choice(options)
        
        # 3. Special handling for troop_priority if present
        if "troop_priority" in mutated and random.random() < self.mutation_rate:
            # Pick two positions and swap them
            if len(mutated["troop_priority"]) >= 2:
                idx1 = random.randint(0, len(mutated["troop_priority"]) - 1)
                idx2 = random.randint(0, len(mutated["troop_priority"]) - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, len(mutated["troop_priority"]) - 1)
                    
                mutated["troop_priority"][idx1], mutated["troop_priority"][idx2] = \
                    mutated["troop_priority"][idx2], mutated["troop_priority"][idx1]
        
        return mutated

    def generate_unique_strategies(self, base_strategies: Dict[str, Dict], 
                                   count: int) -> List[Dict]:
        """
        Generate a diverse set of new strategies by mutating base strategies.
        Useful for creating an initial diverse population.
        
        Args:
            base_strategies: Dictionary of base strategies to start from
            count: Number of new strategies to generate
            
        Returns:
            List of new diverse strategies
        """
        new_strategies = []
        strategies_list = list(base_strategies.values())
        
        for i in range(count):
            # Pick a random base strategy
            base = random.choice(strategies_list)
            
            # Create a highly mutated version
            strategy = copy.deepcopy(base)
            
            # Apply multiple mutations
            for _ in range(3):  # Apply 3 rounds of mutation
                strategy = self._mutate(strategy)
            
            # Ensure it's quite different by forcing some parameters to change
            forced_changes = random.sample(list(self.param_ranges.keys()), 
                                          k=min(3, len(self.param_ranges)))
            
            for param_path in forced_changes:
                min_val, max_val, step = self.param_ranges[param_path]
                
                # Parse the parameter path
                path_parts = param_path.split(".")
                
                # Navigate to the right part of the dictionary
                target = strategy
                for part in path_parts[:-1]:
                    if part.isdigit():
                        part = int(part)
                    target = target[part]
                
                # Get the final key
                final_key = path_parts[-1]
                if final_key.isdigit():
                    final_key = int(final_key)
                
                # Apply a significant change
                if isinstance(step, int):
                    new_val = random.randint(min_val, max_val)
                    target[final_key] = int(new_val)
                else:
                    new_val = min_val + random.random() * (max_val - min_val)
                    target[final_key] = new_val
            
            # Also change lane preference
            if "lane_preference" in strategy:
                strategy["lane_preference"] = random.choice(self.categorical_options["lane_preference"])
            
            # Generate new ID and reset metrics
            strategy["id"] = f"diverse_{str(uuid.uuid4())[:8]}"
            strategy["name"] = f"diverse_{i+1}"
            strategy["metrics"] = {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "avg_tower_health": 0,
                "win_rate": 0
            }
            
            new_strategies.append(strategy)
        
        return new_strategies


if __name__ == "__main__":
    # Example usage
    evolver = StrategyEvolver()
    
    # Sample strategies
    strategies = {
        "strat1": {
            "id": "strat1",
            "name": "aggressive",
            "lane_preference": "right",
            "elixir_thresholds": {
                "min_deploy": 3,
                "save_threshold": 7,
                "emergency_threshold": 2
            },
            "position_settings": {
                "x_range": [-20, 20],
                "y_default": 45,
                "defensive_y": 20
            },
            "defensive_trigger": 0.4,
            "counter_weights": {
                "air_vs_ground": 1.5,
                "splash_vs_group": 1.7,
                "tank_priority": 1.2
            },
            "troop_priority": ["dragon", "wizard", "valkyrie", "knight"],
            "metrics": {
                "games_played": 10,
                "wins": 8,
                "losses": 2,
                "avg_tower_health": 7500,
                "win_rate": 0.8
            }
        },
        "strat2": {
            "id": "strat2",
            "name": "defensive",
            "lane_preference": "left",
            "elixir_thresholds": {
                "min_deploy": 5,
                "save_threshold": 8,
                "emergency_threshold": 3
            },
            "position_settings": {
                "x_range": [-15, 15],
                "y_default": 30,
                "defensive_y": 15
            },
            "defensive_trigger": 0.7,
            "counter_weights": {
                "air_vs_ground": 1.2,
                "splash_vs_group": 2.0,
                "tank_priority": 1.5
            },
            "troop_priority": ["knight", "archer", "wizard", "dragon"],
            "metrics": {
                "games_played": 10,
                "wins": 5,
                "losses": 5,
                "avg_tower_health": 5000,
                "win_rate": 0.5
            }
        }
    }
    
    # Generate evolved offspring
    offspring = evolver.evolve_population(strategies, 1)
    
    # Print one of the offspring
    if offspring:
        print("Example evolved strategy:")
        print(f"ID: {offspring[0]['id']}")
        print(f"Name: {offspring[0]['name']}")
        print(f"Lane: {offspring[0]['lane_preference']}")
        print(f"Min deploy elixir: {offspring[0]['elixir_thresholds']['min_deploy']}")
        print(f"Defensive trigger: {offspring[0]['defensive_trigger']}")
        if "troop_priority" in offspring[0]:
            print(f"Troop priority: {offspring[0]['troop_priority']}")