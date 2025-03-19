import random
import copy
import uuid
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

class StrategyEvolver:
    """
    Enhanced genetic algorithm for tower defense strategy evolution.
    Creates robust, adaptive strategies through advanced selection and mutation.
    """
    
    def __init__(self, mutation_rate=0.25, crossover_rate=0.75, tournament_size=4,
                 elite_preservation=0.1, diversity_weight=0.2):
        """
        Initialize the strategy evolver with enhanced parameters.
        
        Args:
            mutation_rate: Base probability of mutation for each parameter
            crossover_rate: Probability of crossover between parents
            tournament_size: Number of strategies to consider in tournament selection
            elite_preservation: Percentage of top performers to preserve unchanged
            diversity_weight: Weight given to maintaining strategic diversity
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_preservation = elite_preservation
        self.diversity_weight = diversity_weight
        
        # Parameter mutation ranges with finer control
        self.param_ranges = {
            # Elixir management parameters
            "elixir_thresholds.min_deploy": (2.5, 5.5, 0.25),     # (min, max, step)
            "elixir_thresholds.save_threshold": (6.0, 9.0, 0.5),
            "elixir_thresholds.emergency_threshold": (1.0, 3.0, 0.25),
            
            # Position parameters - optimized from best performers
            "position_settings.x_range.0": (-25, -5, 2),      # Left x-range bound
            "position_settings.x_range.1": (5, 25, 2),        # Right x-range bound
            "position_settings.y_default": (35, 49, 1),       # Default y position
            "position_settings.y_aggressive": (42, 49, 1),    # Aggressive y position
            "position_settings.defensive_y": (15, 35, 2),     # Defensive position
            
            # Strategy control thresholds
            "defensive_trigger": (0.45, 0.85, 0.05),          # When to go defensive
            "aggression_trigger": (0.10, 0.30, 0.05),         # When to be aggressive

            # Counter weights for troop selection
            "counter_weights.air_vs_ground": (0.8, 3.0, 0.1),
            "counter_weights.splash_vs_group": (1.0, 3.0, 0.1),
            "counter_weights.tank_priority": (0.8, 2.5, 0.1),
            "counter_weights.range_bonus": (1.0, 2.0, 0.1)
        }
        
        # Categorical parameter options
        self.categorical_options = {
            "lane_preference": ["left", "right", "center", "split", "adaptive"]
        }
        
        # Track parameter distributions for directed evolution
        self.param_distributions = {}
        self.generation = 0
        
        # Troop types for strategic diversity
        self.troop_types = {
            "archer": {"type": "ground", "targets": ["air", "ground"], "range": "high"},
            "giant": {"type": "ground", "targets": ["ground"], "range": "melee"},
            "dragon": {"type": "air", "targets": ["air", "ground"], "range": "medium"},
            "balloon": {"type": "air", "targets": ["ground"], "range": "melee"},
            "prince": {"type": "ground", "targets": ["ground"], "range": "melee"},
            "barbarian": {"type": "ground", "targets": ["ground"], "range": "melee"},
            "knight": {"type": "ground", "targets": ["ground"], "range": "melee"},
            "minion": {"type": "air", "targets": ["air", "ground"], "range": "medium"},
            "skeleton": {"type": "ground", "targets": ["ground"], "range": "melee"},
            "wizard": {"type": "ground", "targets": ["air", "ground"], "range": "high"},
            "valkyrie": {"type": "ground", "targets": ["ground"], "range": "melee"},
            "musketeer": {"type": "ground", "targets": ["air", "ground"], "range": "high"}
        }
        
    def evolve_population(self, strategies: Dict[str, Dict], num_offspring: int = 4,
                          current_generation: int = None) -> List[Dict]:
        """
        Create a new generation of strategies through directed evolution.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy
            num_offspring: Number of new strategies to create
            current_generation: Current generation number
            
        Returns:
            List of new evolved strategies
        """
        # Update generation counter
        if current_generation is not None:
            self.generation = current_generation
        else:
            self.generation += 1
            
        # Convert strategies dict to list, filtering out those with no games
        strategy_list = [s for s in strategies.values() 
                        if s.get("metrics", {}).get("games_played", 0) > 0]
        
        if len(strategy_list) < 2:
            print("Not enough strategies with games played for evolution")
            return []
        
        # Update parameter distributions from successful strategies
        self.update_parameter_distributions(strategy_list)
        
        # Identify elite strategies to preserve
        elite_strategies = self.select_elite_strategies(strategy_list)
        
        # Create offspring
        offspring = []
        
        # First, add elite strategies with small mutations if desired
        if elite_strategies:
            print(f"Adding {len(elite_strategies)} elite strategies with minor refinements")
            for elite in elite_strategies:
                # Create a lightly mutated version of the elite
                refined_elite = self._refine_elite_strategy(elite)
                offspring.append(refined_elite)
                
                # If we've generated enough offspring, stop
                if len(offspring) >= num_offspring:
                    break
        
        # Then add newly evolved strategies
        remaining = max(0, num_offspring - len(offspring))
        if remaining > 0:
            print(f"Evolving {remaining} new strategies")
            
            # Use diversity-aware parent selection for remaining strategies
            for _ in range(remaining):
                # Select parents using tournament selection
                parent1 = self._tournament_selection(strategy_list)
                parent2 = self._diversity_aware_selection(strategy_list, parent1)
                
                # Create child through crossover and mutation
                if random.random() < self.crossover_rate:
                    # Crossover
                    child = self._crossover(parent1, parent2)
                else:
                    # Just use one parent as template
                    child = copy.deepcopy(parent1)
                
                # Apply mutations with generation-dependent rate
                child = self._mutate(child)
                
                # Reset metrics and generate new ID
                child["id"] = f"gen_{self.generation}_{str(uuid.uuid4())[:6]}"
                child["name"] = f"gen_{self.generation}"
                child["metrics"] = {
                    "games_played": 0,
                    "wins": 0,
                    "losses": 0,
                    "avg_tower_health": 0,
                    "win_rate": 0
                }
                child["generation"] = self.generation
                
                offspring.append(child)
            
        print(f"Created {len(offspring)} new evolved strategies for generation {self.generation}")
        return offspring
    
    def update_parameter_distributions(self, strategies: List[Dict]) -> None:
        """
        Update parameter distributions based on successful strategies.
        This guides future mutations toward promising parameter values.
        
        Args:
            strategies: List of strategies
        """
        # Skip if no strategies
        if not strategies:
            return
            
        # Weight strategies by win rate
        weighted_strategies = []
        for strategy in strategies:
            metrics = strategy.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            games_played = metrics.get("games_played", 0)
            
            # Only consider strategies with sufficient games
            if games_played >= 5:
                # Use confidence-adjusted win rate as weight
                # Add small bonus for more games played (max +0.1 for 20+ games)
                experience_bonus = min(0.1, games_played / 200)
                weight = win_rate + experience_bonus
                weighted_strategies.append((strategy, weight))
        
        # Skip if no strategies with sufficient games
        if not weighted_strategies:
            return
            
        # Normalize weights
        total_weight = sum(weight for _, weight in weighted_strategies)
        if total_weight <= 0:
            return
            
        # Track parameter values and their weights
        param_values = defaultdict(list)
        
        # Process each strategy
        for strategy, weight in weighted_strategies:
            # Process numeric parameters
            for param_path in self.param_ranges.keys():
                # Parse the parameter path
                parts = param_path.split('.')
                value = None
                
                # Navigate the nested dictionary
                current = strategy
                for i, part in enumerate(parts):
                    if i == len(parts) - 1 and part.isdigit():
                        # Handle array index
                        idx = int(part)
                        parent_key = parts[i-1]
                        if parent_key in current and len(current[parent_key]) > idx:
                            value = current[parent_key][idx]
                    elif part in current:
                        if i == len(parts) - 1:
                            value = current[part]
                        else:
                            current = current[part]
                
                # Record value and weight if found
                if value is not None:
                    param_values[param_path].append((value, weight))
            
            # Process categorical parameters
            for param, options in self.categorical_options.items():
                if param in strategy:
                    value = strategy[param]
                    param_values[param].append((value, weight))
                    
        # Calculate parameter distributions
        for param, values in param_values.items():
            if values:
                # For numerical parameters, calculate weighted average and standard deviation
                if param in self.param_ranges:
                    values_array = np.array([v for v, _ in values])
                    weights_array = np.array([w for _, w in values])
                    weights_array = weights_array / weights_array.sum()  # Normalize weights
                    
                    # Weighted average
                    avg = np.average(values_array, weights=weights_array)
                    # Weighted standard deviation
                    variance = np.average((values_array - avg)**2, weights=weights_array)
                    std_dev = math.sqrt(variance)
                    
                    self.param_distributions[param] = {
                        "mean": avg,
                        "std_dev": std_dev,
                        "values": values
                    }
                    
                # For categorical parameters, calculate frequency distribution
                elif param in self.categorical_options:
                    freq = defaultdict(float)
                    for value, weight in values:
                        freq[value] += weight
                    
                    # Normalize to get probabilities
                    total = sum(freq.values())
                    if total > 0:
                        for key in freq:
                            freq[key] /= total
                    
                    self.param_distributions[param] = {
                        "frequencies": dict(freq)
                    }
    
    def select_elite_strategies(self, strategies: List[Dict]) -> List[Dict]:
        """
        Select top strategies to preserve with minimal mutation.
        
        Args:
            strategies: List of strategies
            
        Returns:
            List of elite strategies
        """
        # Sort strategies by win rate
        sorted_strategies = sorted(
            strategies,
            key=lambda s: s.get("metrics", {}).get("win_rate", 0),
            reverse=True
        )
        
        # Select top percentage as elites
        elite_count = max(1, int(len(strategies) * self.elite_preservation))
        return sorted_strategies[:elite_count]
    
    def _refine_elite_strategy(self, strategy: Dict) -> Dict:
        """
        Create a slightly refined version of an elite strategy.
        Uses lighter mutation to make small improvements.
        
        Args:
            strategy: Elite strategy to refine
            
        Returns:
            Refined strategy
        """
        # Create a deep copy
        refined = copy.deepcopy(strategy)
        
        # Apply very mild mutations (1/4 the normal rate)
        original_rate = self.mutation_rate
        self.mutation_rate *= 0.25
        refined = self._mutate(refined)
        self.mutation_rate = original_rate
        
        # Generate new ID but preserve generation marker for tracking lineage
        gen_prefix = f"gen_{self.generation}"
        if "id" in strategy:
            old_id = strategy["id"]
            if "_" in old_id:
                parts = old_id.split("_")
                if len(parts) >= 3:
                    # Preserve the original generation number in the ID
                    gen_prefix = f"gen_{parts[1]}"
        
        refined["id"] = f"{gen_prefix}_{str(uuid.uuid4())[:6]}"
        refined["name"] = gen_prefix
        refined["metrics"] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "avg_tower_health": 0,
            "win_rate": 0
        }
        refined["generation"] = self.generation
        
        # Add elite marker for tracking
        refined["refined_from"] = strategy.get("id", "unknown")
        
        return refined
    
    def _tournament_selection(self, strategies: List[Dict]) -> Dict:
        """
        Select a strategy using tournament selection based on win rate.
        
        Args:
            strategies: List of strategies to select from
            
        Returns:
            Selected strategy
        """
        # Adjust tournament size if we have fewer strategies
        actual_tournament_size = min(self.tournament_size, len(strategies))
        
        # Randomly select strategies for tournament
        tournament = random.sample(strategies, actual_tournament_size)
        
        # Find the best strategy in the tournament
        best_strategy = None
        best_fitness = -1
        
        for strategy in tournament:
            # Calculate fitness (weighted combination of win rate and games played)
            metrics = strategy.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            games_played = metrics.get("games_played", 0)
            
            # Fitness function: win_rate with a small bonus for experience
            fitness = win_rate + (0.01 * min(games_played, 20))
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_strategy = strategy
        
        return best_strategy
    
    def _diversity_aware_selection(self, strategies: List[Dict], excluded_strategy: Dict) -> Dict:
        """
        Select a strategy that provides good genetic diversity compared to the first parent.
        
        Args:
            strategies: List of strategies to select from
            excluded_strategy: Strategy to ensure diversity from
            
        Returns:
            Selected diverse strategy
        """
        # Get a fresh tournament of candidates
        actual_tournament_size = min(self.tournament_size * 2, len(strategies))
        tournament = random.sample(strategies, actual_tournament_size)
        
        # Calculate diversity scores
        best_strategy = None
        best_score = -1
        
        for strategy in tournament:
            if strategy["id"] == excluded_strategy["id"]:
                continue
                
            # Calculate base fitness
            metrics = strategy.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            games_played = metrics.get("games_played", 0)
            base_fitness = win_rate + (0.01 * min(games_played, 20))
            
            # Calculate diversity score
            diversity = self._calculate_diversity(strategy, excluded_strategy)
            
            # Combine win rate and diversity
            combined_score = (1 - self.diversity_weight) * base_fitness + self.diversity_weight * diversity
            
            if combined_score > best_score:
                best_score = combined_score
                best_strategy = strategy
        
        # If no suitable strategy found, fall back to tournament selection
        if best_strategy is None:
            strategies_copy = [s for s in strategies if s["id"] != excluded_strategy["id"]]
            if strategies_copy:
                return self._tournament_selection(strategies_copy)
            else:
                # Just return another copy of the excluded strategy if nothing else available
                return excluded_strategy
                
        return best_strategy
    
    def _calculate_diversity(self, strategy1: Dict, strategy2: Dict) -> float:
        """
        Calculate diversity score between two strategies (0=identical, 1=completely different).
        
        Args:
            strategy1: First strategy
            strategy2: Second strategy
            
        Returns:
            Diversity score from 0 to 1
        """
        # Initialize diversity metrics
        diff_points = 0
        total_points = 0
        
        # Compare numeric parameters with parameter ranges
        for param_path, (min_val, max_val, _) in self.param_ranges.items():
            value1 = self._get_nested_value(strategy1, param_path)
            value2 = self._get_nested_value(strategy2, param_path)
            
            if value1 is not None and value2 is not None:
                # Calculate normalized difference
                param_range = max(0.0001, max_val - min_val)  # Avoid division by zero
                norm_diff = min(1.0, abs(value1 - value2) / param_range)
                diff_points += norm_diff
                total_points += 1
        
        # Compare categorical parameters
        for param, options in self.categorical_options.items():
            value1 = strategy1.get(param)
            value2 = strategy2.get(param)
            
            if value1 is not None and value2 is not None:
                if value1 != value2:
                    diff_points += 1
                total_points += 1
        
        # Compare troop priorities if present
        if "troop_priority" in strategy1 and "troop_priority" in strategy2:
            # Calculate Jaccard distance for the first few troops
            n = min(4, len(strategy1["troop_priority"]), len(strategy2["troop_priority"]))
            set1 = set(strategy1["troop_priority"][:n])
            set2 = set(strategy2["troop_priority"][:n])
            
            # Jaccard distance: 1 - (intersection size / union size)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            if union > 0:
                diff_points += 1 - (intersection / union)
            total_points += 1
            
            # Check for order differences in shared troops
            ordered_diff = 0
            for i in range(n):
                if i < len(strategy1["troop_priority"]) and i < len(strategy2["troop_priority"]):
                    if strategy1["troop_priority"][i] != strategy2["troop_priority"][i]:
                        ordered_diff += 1
            if n > 0:
                diff_points += ordered_diff / n
                total_points += 1
        
        # Calculate final diversity score
        diversity = diff_points / total_points if total_points > 0 else 0.5
        return diversity
    
    def _get_nested_value(self, strategy: Dict, param_path: str) -> Optional[Any]:
        """
        Get a value from a nested dictionary structure using a dotted path.
        
        Args:
            strategy: Strategy dictionary
            param_path: Parameter path with dot notation (e.g., "elixir_thresholds.min_deploy")
            
        Returns:
            Parameter value if found, None otherwise
        """
        parts = param_path.split('.')
        current = strategy
        
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:
                # Handle array index
                idx = int(part)
                prev_part = parts[i-1]
                if prev_part in current and isinstance(current[prev_part], list) and len(current[prev_part]) > idx:
                    return current[prev_part][idx]
                else:
                    return None
            elif part in current:
                if i == len(parts) - 1:
                    return current[part]
                else:
                    current = current[part]
            else:
                return None
                
        return None
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Create a new strategy by combining two parent strategies.
        Uses more sophisticated crossover techniques.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            New strategy created through crossover
        """
        # Create a deep copy of parent1 as the base
        child = copy.deepcopy(parent1)
        
        # 1. Crossover for nested dictionary parameters using parameter distributions
        for main_key in ["elixir_thresholds", "position_settings", "counter_weights"]:
            if main_key in parent1 and main_key in parent2:
                # For each sub-parameter, intelligently choose or blend values
                for sub_key in parent1[main_key]:
                    if sub_key in parent2[main_key]:
                        # Special handling for x_range which is a list
                        if sub_key == "x_range" and isinstance(parent2[main_key][sub_key], list):
                            # Consider each x-range bound individually
                            for i in range(len(child[main_key][sub_key])):
                                if i < len(parent2[main_key][sub_key]):
                                    # Sometimes blend values, sometimes take directly from a parent
                                    if random.random() < 0.3:  # 30% chance to blend
                                        weight = random.random()  # Blend weight
                                        child[main_key][sub_key][i] = (
                                            parent1[main_key][sub_key][i] * weight +
                                            parent2[main_key][sub_key][i] * (1 - weight)
                                        )
                                    elif random.random() > 0.5:  # 35% chance for parent2
                                        child[main_key][sub_key][i] = parent2[main_key][sub_key][i]
                                    # Otherwise keep parent1's value (35% chance)
                        else:
                            # For other parameters, similar logic
                            if random.random() < 0.3:  # 30% chance to blend
                                weight = random.random()
                                child[main_key][sub_key] = (
                                    parent1[main_key][sub_key] * weight +
                                    parent2[main_key][sub_key] * (1 - weight)
                                )
                            elif random.random() > 0.5:  # 35% chance for parent2
                                child[main_key][sub_key] = parent2[main_key][sub_key]
                            # Otherwise keep parent1's value (35% chance)
        
        # 2. Crossover for top-level parameters
        for key in ["lane_preference", "defensive_trigger"]:
            if key in parent1 and key in parent2 and random.random() > 0.5:
                child[key] = parent2[key]
        
        # 3. Special handling for troop_priority
        if "troop_priority" in parent1 and "troop_priority" in parent2:
            # Choose crossover technique
            crossover_technique = random.choice(["point", "order", "blend"])
            
            if crossover_technique == "point":
                # Traditional crossover point in the list
                crossover_point = random.randint(1, min(len(parent1["troop_priority"]), 
                                                      len(parent2["troop_priority"])) - 1)
                child["troop_priority"] = (
                    parent1["troop_priority"][:crossover_point] + 
                    parent2["troop_priority"][crossover_point:]
                )
                
            elif crossover_technique == "order":
                # Take parent1's troops but in parent2's order
                troops_set = set(parent1["troop_priority"])
                ordered_troops = []
                
                # First add troops in the order they appear in parent2
                for troop in parent2["troop_priority"]:
                    if troop in troops_set:
                        ordered_troops.append(troop)
                        troops_set.remove(troop)
                
                # Then add any remaining troops from parent1
                for troop in parent1["troop_priority"]:
                    if troop in troops_set:
                        ordered_troops.append(troop)
                
                child["troop_priority"] = ordered_troops
                
            else:  # "blend"
                # Take some troops from each parent, prioritizing early positions
                p1_priority = {troop: i for i, troop in enumerate(parent1["troop_priority"])}
                p2_priority = {troop: i for i, troop in enumerate(parent2["troop_priority"])}
                
                # Combine and sort by average priority
                combined = set(parent1["troop_priority"]) | set(parent2["troop_priority"])
                
                # Calculate average priority (lower value = higher priority)
                # If troop only in one parent, use its position there
                troop_priorities = []
                for troop in combined:
                    p1_pos = p1_priority.get(troop, len(parent1["troop_priority"]))
                    p2_pos = p2_priority.get(troop, len(parent2["troop_priority"]))
                    
                    # If in both parents, use average
                    if troop in p1_priority and troop in p2_priority:
                        avg_pos = (p1_pos + p2_pos) / 2
                    else:
                        # If only in one parent, use that position
                        avg_pos = p1_pos if troop in p1_priority else p2_pos
                        
                    troop_priorities.append((troop, avg_pos))
                
                # Sort by priority
                troop_priorities.sort(key=lambda x: x[1])
                
                # Take original number of troops
                child["troop_priority"] = [t[0] for t in troop_priorities[:len(parent1["troop_priority"])]]
            
            # Check for duplicates
            seen = set()
            unique_troops = []
            for troop in child["troop_priority"]:
                if troop not in seen:
                    seen.add(troop)
                    unique_troops.append(troop)
            
            # If we lost troops due to deduplication, add missing ones from a random parent
            source_parent = parent1 if random.random() > 0.5 else parent2
            if len(unique_troops) < len(source_parent["troop_priority"]):
                for troop in source_parent["troop_priority"]:
                    if troop not in seen and len(unique_troops) < len(source_parent["troop_priority"]):
                        unique_troops.append(troop)
                        seen.add(troop)
            
            child["troop_priority"] = unique_troops
            
        return child
    
    def _mutate(self, strategy: Dict) -> Dict:
        """
        Apply intelligent, directed mutations to a strategy based on parameter distributions.
        
        Args:
            strategy: Strategy to mutate
            
        Returns:
            Mutated strategy
        """
        # Create a deep copy to avoid modifying the original
        mutated = copy.deepcopy(strategy)
        
        # Get mutation rate adjusted for generation
        # Early generations: higher mutation for exploration
        # Late generations: lower mutation for exploitation
        adjusted_mutation_rate = self.mutation_rate
        if self.generation > 50:
            adjusted_mutation_rate *= 0.7  # Reduce mutation rate for fine-tuning
        elif self.generation < 10:
            adjusted_mutation_rate *= 1.3  # Higher mutation rate for early exploration
        
        # 1. Mutate numeric parameters with defined ranges
        for param_path, (min_val, max_val, step) in self.param_ranges.items():
            # Determine if this parameter should mutate
            if random.random() > adjusted_mutation_rate:
                continue
            
            # Parse the parameter path
            path_parts = param_path.split(".")
            
            # Navigate to the right part of the dictionary
            target = mutated
            for part in path_parts[:-1]:
                if part.isdigit():  # Handle numeric indices for lists
                    part = int(part)
                if part not in target:
                    # Create missing structure
                    if path_parts[path_parts.index(part) + 1].isdigit():
                        target[part] = []  # Create list for numeric next part
                    else:
                        target[part] = {}  # Create dict otherwise
                target = target[part]
            
            # Get the final key
            final_key = path_parts[-1]
            if final_key.isdigit():  # Handle numeric indices for lists
                final_key = int(final_key)
                
            # Ensure the container can hold this index
            if isinstance(target, list) and len(target) <= final_key:
                # Extend list to accommodate index
                target.extend([0] * (final_key - len(target) + 1))
                
            # Current value or default
            if final_key in target:
                current_val = target[final_key]
            else:
                # Use mid-range as default
                current_val = (min_val + max_val) / 2
            
            # Check if we have distribution data for this parameter
            if param_path in self.param_distributions:
                dist = self.param_distributions[param_path]
                mean = dist.get("mean")
                std_dev = dist.get("std_dev")
                
                if mean is not None and std_dev is not None:
                    # Use Gaussian mutation centered on the mean of successful strategies
                    # with standard deviation influenced by the population
                    mutation_strength = random.gauss(0, 1) * std_dev * 1.5
                    
                    # Interpolate between current value and distribution mean
                    interp_weight = random.random() * 0.7  # 0 to 0.7
                    new_val = current_val * (1 - interp_weight) + mean * interp_weight + mutation_strength
                    
                    # Ensure it stays in range
                    new_val = max(min_val, min(max_val, new_val))
                    
                    # Discretize according to step size
                    if step != 0:
                        new_val = round(new_val / step) * step
                        
                    target[final_key] = new_val
                    continue
            
            # Fallback to standard mutation if no distribution data
            if isinstance(step, int):
                # Integer parameter
                # More conservative mutations for later generations
                if self.generation > 30:
                    new_val = current_val + random.choice([-1, 0, 1]) * step
                else:
                    new_val = current_val + random.randint(-2, 2) * step
                    
                new_val = max(min_val, min(max_val, new_val))
                target[final_key] = int(new_val)
            else:
                                # Float parameter
                # More conservative mutations for later generations
                if self.generation > 30:
                    # Smaller mutations for fine-tuning
                    new_val = current_val + (random.random() * 2 - 1) * step
                else:
                    # Larger mutations for exploration
                    new_val = current_val + (random.random() * 4 - 2) * step
                    
                new_val = max(min_val, min(max_val, new_val))
                target[final_key] = new_val
        
        # 2. Mutate categorical parameters
        for param, options in self.categorical_options.items():
            if random.random() > adjusted_mutation_rate:
                continue
                
            if param in mutated:
                current_value = mutated[param]
                
                # Check if we have distribution data for this parameter
                if param in self.param_distributions:
                    dist = self.param_distributions[param]
                    frequencies = dist.get("frequencies", {})
                    
                    if frequencies:
                        # Use frequency-weighted selection with perturbation
                        # 70% chance to use weighted distribution, 30% to explore randomly
                        if random.random() < 0.7:
                            # Convert frequencies to list of options and weights
                            options_list = list(frequencies.keys())
                            weights_list = list(frequencies.values())
                            
                            # Sample from distribution
                            if sum(weights_list) > 0:
                                mutated[param] = random.choices(options_list, weights=weights_list, k=1)[0]
                            else:
                                mutated[param] = random.choice(options)
                        else:
                            # Random exploration
                            mutated[param] = random.choice(options)
                    else:
                        # No frequency data, use uniform random
                        mutated[param] = random.choice(options)
                else:
                    # No distribution data, use uniform random
                    mutated[param] = random.choice(options)
        
        # 3. Special handling for troop_priority
        if "troop_priority" in mutated and random.random() < adjusted_mutation_rate:
            priority = mutated["troop_priority"]
            
            if len(priority) >= 2:
                # Apply different mutation techniques
                mutation_type = random.choices(
                    ["swap", "shift", "insert", "replace"],
                    weights=[0.4, 0.3, 0.2, 0.1],
                    k=1
                )[0]
                
                if mutation_type == "swap":
                    # Swap two positions
                    pos1 = random.randint(0, len(priority) - 1)
                    pos2 = random.randint(0, len(priority) - 1)
                    while pos1 == pos2 and len(priority) > 1:
                        pos2 = random.randint(0, len(priority) - 1)
                        
                    priority[pos1], priority[pos2] = priority[pos2], priority[pos1]
                    
                elif mutation_type == "shift":
                    # Shift a troop to an earlier position (higher priority)
                    if len(priority) >= 3:
                        from_pos = random.randint(1, len(priority) - 1)
                        to_pos = random.randint(0, from_pos - 1)
                        troop = priority.pop(from_pos)
                        priority.insert(to_pos, troop)
                        
                elif mutation_type == "insert":
                    # Move a random troop to the front
                    if len(priority) >= 3:
                        pos = random.randint(1, len(priority) - 1)
                        troop = priority.pop(pos)
                        priority.insert(0, troop)
                        
                elif mutation_type == "replace":
                    # Replace a low priority troop with one that isn't in the list
                    if len(priority) >= 5:
                        available_troops = [t for t in self.troop_types if t not in priority]
                        if available_troops:
                            pos = random.randint(len(priority) // 2, len(priority) - 1)  # Replace from lower half
                            priority[pos] = random.choice(available_troops)
        
        # 4. Balance troop composition for strategic diversity
        # This ensures we have a mix of air/ground and ranged/melee troops
        if "troop_priority" in mutated and random.random() < adjusted_mutation_rate * 0.5:
            priority = mutated["troop_priority"]
            
            # Check if we have at least one air troop in the top half
            has_air = False
            for i in range(min(4, len(priority))):
                if priority[i] in self.troop_types and self.troop_types[priority[i]]["type"] == "air":
                    has_air = True
                    break
            
            # If no air troops in top positions, promote one
            if not has_air:
                air_troops = [i for i, troop in enumerate(priority) 
                            if troop in self.troop_types and self.troop_types[troop]["type"] == "air"]
                if air_troops:
                    # Pick a random air troop and move it up
                    air_pos = random.choice(air_troops)
                    air_troop = priority.pop(air_pos)
                    priority.insert(random.randint(0, 2), air_troop)
            
            # Similarly ensure we have at least one high-range troop in top positions
            has_range = False
            for i in range(min(4, len(priority))):
                if (priority[i] in self.troop_types and 
                    self.troop_types[priority[i]]["range"] == "high"):
                    has_range = True
                    break
            
            # If no ranged troops in top positions, promote one
            if not has_range:
                range_troops = [i for i, troop in enumerate(priority) 
                               if troop in self.troop_types and self.troop_types[troop]["range"] == "high"]
                if range_troops:
                    # Pick a random ranged troop and move it up
                    range_pos = random.choice(range_troops)
                    range_troop = priority.pop(range_pos)
                    priority.insert(random.randint(0, 2), range_troop)
        
        # 5. Ensure parameter consistency
        # Make sure related parameters make sense together
        if ("elixir_thresholds" in mutated and 
            "min_deploy" in mutated["elixir_thresholds"] and 
            "save_threshold" in mutated["elixir_thresholds"]):
            # Ensure save threshold > min deploy
            min_deploy = mutated["elixir_thresholds"]["min_deploy"]
            save_threshold = mutated["elixir_thresholds"]["save_threshold"]
            if save_threshold <= min_deploy:
                mutated["elixir_thresholds"]["save_threshold"] = min_deploy + random.uniform(1.0, 3.0)
                
        if ("position_settings" in mutated and 
            "defensive_y" in mutated["position_settings"] and
            "y_default" in mutated["position_settings"]):
            # Ensure defensive_y < y_default (defensive is closer to our tower)
            if mutated["position_settings"]["defensive_y"] > mutated["position_settings"]["y_default"]:
                # Swap them
                temp = mutated["position_settings"]["defensive_y"]
                mutated["position_settings"]["defensive_y"] = mutated["position_settings"]["y_default"]
                mutated["position_settings"]["y_default"] = temp
                
        return mutated

    def generate_diverse_population(self, base_strategies: Dict[str, Dict], 
                                   count: int) -> List[Dict]:
        """
        Generate a diverse set of strategies by mutating base strategies.
        Used to create diverse initial population.
        
        Args:
            base_strategies: Dictionary of base strategies
            count: Number of new strategies to create
            
        Returns:
            List of diverse strategies
        """
        new_strategies = []
        strategies_list = list(base_strategies.values())
        
        if not strategies_list:
            print("No base strategies provided for diversification")
            return []
        
        print(f"Generating {count} diverse strategies")
        
        # Create strategies with high diversity
        for i in range(count):
            # Pick a random base strategy
            base = random.choice(strategies_list)
            
            # Create a highly mutated version
            strategy = copy.deepcopy(base)
            
            # Apply multiple rounds of mutation
            for _ in range(3):  # Apply 3 rounds of strong mutation
                # Use a temporarily higher mutation rate
                original_rate = self.mutation_rate
                self.mutation_rate = min(0.9, self.mutation_rate * 2)
                strategy = self._mutate(strategy)
                self.mutation_rate = original_rate
            
            # Force changes to specific parameters for diversity
            self._force_parameter_diversity(strategy, i % 4)  # Use modulo to cycle through diversity types
            
            # Generate new ID 
            strategy["id"] = f"diverse_{self.generation}_{str(uuid.uuid4())[:6]}"
            strategy["name"] = f"diverse_{self.generation}"
            
            # Reset metrics
            strategy["metrics"] = {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "avg_tower_health": 0,
                "win_rate": 0
            }
            strategy["generation"] = self.generation
            
            new_strategies.append(strategy)
        
        return new_strategies
    
    def _force_parameter_diversity(self, strategy: Dict, diversity_type: int) -> None:
        """
        Force diversity by making significant changes to specific parameters.
        
        Args:
            strategy: Strategy to diversify
            diversity_type: Type of diversity to enforce (0-3)
        """
        # Different diversity types focus on different aspects
        if diversity_type == 0:
            # Aggressive type
            if "lane_preference" in self.categorical_options:
                strategy["lane_preference"] = random.choice(["right", "split"])
            if "elixir_thresholds" in strategy:
                strategy["elixir_thresholds"]["min_deploy"] = random.uniform(2.5, 3.5)
                strategy["elixir_thresholds"]["save_threshold"] = random.uniform(6.0, 7.0)
            if "position_settings" in strategy:
                strategy["position_settings"]["y_default"] = random.randint(42, 49)
            if "defensive_trigger" in strategy:
                strategy["defensive_trigger"] = random.uniform(0.45, 0.55)
            
        elif diversity_type == 1:
            # Defensive type
            if "lane_preference" in self.categorical_options:
                strategy["lane_preference"] = random.choice(["center", "left"])
            if "elixir_thresholds" in strategy:
                strategy["elixir_thresholds"]["min_deploy"] = random.uniform(4.0, 5.5)
                strategy["elixir_thresholds"]["save_threshold"] = random.uniform(7.0, 9.0)
            if "position_settings" in strategy:
                strategy["position_settings"]["y_default"] = random.randint(35, 42)
            if "defensive_trigger" in strategy:
                strategy["defensive_trigger"] = random.uniform(0.65, 0.85)
            
        elif diversity_type == 2:
            # Balanced type
            if "lane_preference" in self.categorical_options:
                strategy["lane_preference"] = random.choice(["center", "adaptive"])
            if "elixir_thresholds" in strategy:
                strategy["elixir_thresholds"]["min_deploy"] = random.uniform(3.5, 4.5)
                strategy["elixir_thresholds"]["save_threshold"] = random.uniform(6.5, 8.0)
            if "position_settings" in strategy:
                strategy["position_settings"]["y_default"] = random.randint(38, 45)
            if "defensive_trigger" in strategy:
                strategy["defensive_trigger"] = random.uniform(0.55, 0.75)
                
        else:  # diversity_type == 3
            # Experimental/extreme type
            if "lane_preference" in self.categorical_options:
                strategy["lane_preference"] = random.choice(["split", "adaptive"])
            if "elixir_thresholds" in strategy:
                # Either very conservative or very aggressive elixir use
                if random.random() > 0.5:
                    strategy["elixir_thresholds"]["min_deploy"] = random.uniform(2.5, 3.0)
                    strategy["elixir_thresholds"]["save_threshold"] = random.uniform(5.0, 6.0)
                else:
                    strategy["elixir_thresholds"]["min_deploy"] = random.uniform(5.0, 5.5)
                    strategy["elixir_thresholds"]["save_threshold"] = random.uniform(8.0, 9.0)
            if "position_settings" in strategy:
                # Either very aggressive or very defensive positioning
                if random.random() > 0.5:
                    strategy["position_settings"]["y_default"] = random.randint(45, 49)
                else:
                    strategy["position_settings"]["y_default"] = random.randint(35, 38)
            if "defensive_trigger" in strategy:
                # Either wait until very damaged or switch early
                if random.random() > 0.5:
                    strategy["defensive_trigger"] = random.uniform(0.45, 0.50)
                else:
                    strategy["defensive_trigger"] = random.uniform(0.80, 0.85)
        
        # Regardless of type, ensure troop diversity
        if "troop_priority" in strategy and len(strategy["troop_priority"]) > 4:
            # Different diversity types favor different troop compositions
            if diversity_type == 0:  # Aggressive
                # Prioritize high damage troops
                primary_choices = ["dragon", "wizard", "prince", "musketeer"]
            elif diversity_type == 1:  # Defensive
                # Prioritize defensive troops
                primary_choices = ["wizard", "valkyrie", "archer", "musketeer"]
            elif diversity_type == 2:  # Balanced
                # Balanced mix
                primary_choices = ["dragon", "wizard", "knight", "archer"]
            else:  # Experimental
                # Unusual combinations
                primary_choices = ["minion", "dragon", "barbarian", "skeleton"]
                
            # Ensure primary choices that exist in troop_types are at the front
            valid_primary = [t for t in primary_choices if t in self.troop_types]
            remaining = [t for t in strategy["troop_priority"] if t not in valid_primary]
            
            # Combine, keeping original length
            orig_len = len(strategy["troop_priority"])
            strategy["troop_priority"] = valid_primary + remaining
            strategy["troop_priority"] = strategy["troop_priority"][:orig_len]


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
    offspring = evolver.evolve_population(strategies, 2)
    
    # Print evolved strategies
    if offspring:
        print("\nEvolved strategies:")
        for i, child in enumerate(offspring):
            print(f"\nStrategy {i+1}:")
            print(f"ID: {child['id']}")
            print(f"Name: {child['name']}")
            print(f"Lane preference: {child['lane_preference']}")
            print(f"Min deploy elixir: {child['elixir_thresholds']['min_deploy']}")
            print(f"Defensive trigger: {child['defensive_trigger']}")
            if "troop_priority" in child:
                print(f"Troop priority: {child['troop_priority']}")