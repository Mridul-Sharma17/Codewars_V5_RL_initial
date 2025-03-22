"""
Tower Defense RL - Strategy Generator

This module handles the generation of strategy files through genetic algorithms.
"""

import os
import random
import copy
import uuid
import logging
from typing import Dict, List, Any

from config import *
from adaptive_templates import AdaptiveTemplates

logger = logging.getLogger('strategy_generator')

class StrategyGenerator:
    """Generates and evolves tower defense strategies."""
    
    def __init__(self):
        """Initialize the strategy generator."""
        self.templates = AdaptiveTemplates()
        
    def generate_initial_strategies(self, count: int) -> Dict[str, Dict[str, Any]]:
        """
        Generate initial population of strategies.
        
        Args:
            count: Number of strategies to generate
            
        Returns:
            Dictionary of strategy_id -> strategy_data
        """
        strategies = {}
        
        # Calculate how many of each template type to generate based on weights
        template_counts = {}
        remaining = count
        
        for template_type, weight in TEMPLATE_WEIGHTS.items():
            if template_type == "counter_picker":
                # Always include at least one counter picker strategy
                template_counts[template_type] = max(1, int(count * weight))
            else:
                template_counts[template_type] = int(count * weight)
            remaining -= template_counts[template_type]
        
        # Distribute any remaining strategies
        while remaining > 0:
            template_type = random.choice(list(TEMPLATE_WEIGHTS.keys()))
            template_counts[template_type] += 1
            remaining -= 1
        
        # Generate strategies for each template type
        for template_type, template_count in template_counts.items():
            for _ in range(template_count):
                strategy_id = f"{template_type}_{uuid.uuid4().hex[:8]}"
                
                # Generate random parameters for this template type
                parameters = self._generate_random_parameters(template_type)
                
                # Create the strategy
                strategy = {
                    "id": strategy_id,
                    "name": f"{template_type.replace('_', ' ').title()} {uuid.uuid4().hex[:4]}",
                    "template_type": template_type,
                    "parameters": parameters,
                    "generation": 0,
                    "parent_ids": [],
                    "is_static": False,
                    "metrics": {
                        "games_played": 0,
                        "wins": 0,
                        "losses": 0,
                        "avg_tower_health": 0,
                        "win_rate": 0.0,
                        "wilson_score": 0.0,
                        "elite_performance": 0.0,
                        "versatility": 0.0
                    }
                }
                
                # Generate the strategy file
                strategy["source_file"] = self._generate_strategy_file(strategy)
                strategies[strategy_id] = strategy
        
        logger.info(f"Generated {len(strategies)} initial strategies")
        return strategies
    
    def evolve_population(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        fitness_scores: Dict[str, float],
        generation: int,
        population_size: int,
        elite_count: int = 2,
        tournament_size: int = 3,
        crossover_probability: float = 0.7,
        mutation_rate: float = 0.3,
        retention_rate: float = 1.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evolve the population to produce the next generation.
        
        Args:
            strategies: Current population of strategies
            fitness_scores: Fitness scores for each strategy
            generation: Current generation number
            population_size: Desired population size
            elite_count: Number of top strategies to preserve unchanged
            tournament_size: Size of tournaments for selection
            crossover_probability: Probability of crossover vs. reproduction
            mutation_rate: Probability of mutation for each parameter
            retention_rate: Fraction of population to retain (culling)
            
        Returns:
            New population of strategies
        """
        # Sort strategies by fitness score
        sorted_strategies = sorted(
            [s for s_id, s in strategies.items() if not s.get("is_static", False)],
            key=lambda s: fitness_scores.get(s["id"], 0),
            reverse=True
        )
        
        # Calculate how many to retain
        retain_count = max(elite_count, int(len(sorted_strategies) * retention_rate))
        
        # Create new population
        new_population = {}
        
        # First, copy all static strategies
        for strategy_id, strategy in strategies.items():
            if strategy.get("is_static", False):
                new_population[strategy_id] = copy.deepcopy(strategy)
        
        # Then add elites
        elites = sorted_strategies[:elite_count]
        for elite in elites:
            if elite["id"] not in new_population:
                new_population[elite["id"]] = copy.deepcopy(elite)
        
        # Calculate how many new strategies to generate
        new_count = population_size - len(new_population)
        
        # Generate new strategies through selection, crossover and mutation
        parents_pool = sorted_strategies[:retain_count]
        
        while len(new_population) < population_size:
            # Selection
            parent1 = self._tournament_selection(parents_pool, fitness_scores, tournament_size)
            
            # Decide between crossover and reproduction
            if random.random() < crossover_probability and len(parents_pool) > 1:
                # Crossover
                parent2 = self._tournament_selection(
                    [p for p in parents_pool if p["id"] != parent1["id"]], 
                    fitness_scores, 
                    tournament_size
                )
                child = self._crossover(parent1, parent2)
                child["parent_ids"] = [parent1["id"], parent2["id"]]
            else:
                # Reproduction
                child = copy.deepcopy(parent1)
                child["parent_ids"] = [parent1["id"]]
            
            # Mutation
            self._mutate(child, mutation_rate)
            
            # Update child metadata
            child_id = f"{child['template_type']}_{uuid.uuid4().hex[:8]}"
            child["id"] = child_id
            child["name"] = f"{child['template_type'].replace('_', ' ').title()} {uuid.uuid4().hex[:4]}"
            child["generation"] = generation + 1
            
            # Reset metrics
            child["metrics"] = {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "avg_tower_health": 0,
                "win_rate": 0.0,
                "wilson_score": 0.0,
                "elite_performance": 0.0,
                "versatility": 0.0
            }
            
            # Generate the strategy file
            child["source_file"] = self._generate_strategy_file(child)
            
            # Add to population
            new_population[child_id] = child
        
        logger.info(f"Generated {len(new_population) - len(elites) - len([s for s in strategies.values() if s.get('is_static', False)])} new strategies for generation {generation + 1}")
        return new_population
    
    def _generate_random_parameters(self, template_type: str) -> Dict[str, Any]:
        """Generate random parameters for a given template type."""
        if template_type == "counter_picker":
            return {
                # Troop selection parameters
                "counter_weights": {
                    "air": random.uniform(0.5, 3.0),
                    "ground": random.uniform(0.5, 3.0),
                    "splash": random.uniform(0.5, 3.0),
                    "tank": random.uniform(0.5, 3.0),
                    "swarm": random.uniform(0.5, 3.0)
                },
                "troop_to_category": {
                    "Archer": ["ground", "ranged"],
                    "Giant": ["ground", "tank"],
                    "Dragon": ["air", "splash"],
                    "Balloon": ["air", "building-targeting"],
                    "Prince": ["ground", "tank"],
                    "Barbarian": ["ground", "swarm"],
                    "Knight": ["ground", "tank"],
                    "Minion": ["air", "swarm"],
                    "Skeleton": ["ground", "swarm"],
                    "Wizard": ["ground", "splash"],
                    "Valkyrie": ["ground", "splash"],
                    "Musketeer": ["ground", "ranged"]
                },
                "category_counters": {
                    "air": ["ground", "ranged"],
                    "ground": ["air", "splash"],
                    "tank": ["swarm", "building-targeting"],
                    "swarm": ["splash"],
                    "ranged": ["tank"],
                    "splash": ["ranged", "building-targeting"],
                    "building-targeting": ["swarm"]
                },
                # Deployment parameters
                "deploy_distance": random.uniform(5, 20),
                "air_deploy_distance": random.uniform(0, 10),
                "ground_deploy_distance": random.uniform(0, 15),
                "lane_preference": random.uniform(-20, 20),
                "elixir_threshold": random.uniform(4, 8),
                "aggressive_threshold": random.uniform(0.3, 0.7)
            }
        elif template_type == "resource_manager":
            return {
                # Elixir management parameters
                "early_game_threshold": random.randint(30, 60),
                "mid_game_threshold": random.randint(90, 120),
                "early_elixir_threshold": random.uniform(5, 9),
                "mid_elixir_threshold": random.uniform(4, 8),
                "late_elixir_threshold": random.uniform(3, 7),
                "elixir_advantage_threshold": random.uniform(1, 3),
                "troop_value_coefficients": {
                    "health": random.uniform(0.01, 0.1),
                    "damage": random.uniform(0.1, 1.0),
                    "speed": random.uniform(0.5, 2.0),
                    "attack_range": random.uniform(0.2, 1.0),
                    "splash_range": random.uniform(1.0, 3.0)
                },
                # Deployment positioning
                "deploy_distance": random.uniform(5, 20),
                "spread_factor": random.uniform(5, 20),
                "lane_preference": random.uniform(-20, 20)
            }
        elif template_type == "lane_adaptor":
            return {
                # Lane analysis parameters
                "lane_width": random.uniform(10, 25),
                "lane_memory_factor": random.uniform(0.6, 0.9),
                "lane_pressure_threshold": random.uniform(0.4, 0.8),
                # Response parameters
                "defensive_distance": random.uniform(10, 20),
                "offensive_distance": random.uniform(0, 10),
                "grouping_factor": random.uniform(0.0, 1.0),
                "counter_deploy_ratio": random.uniform(0.5, 1.5),
                # Troop selection
                "troop_role_weights": {
                    "tank": random.uniform(0.5, 2.0),
                    "support": random.uniform(0.5, 2.0),
                    "swarm": random.uniform(0.5, 2.0),
                    "splasher": random.uniform(0.5, 2.0)
                },
                "troop_roles": {
                    "Giant": "tank",
                    "Knight": "tank",
                    "Prince": "tank",
                    "Barbarian": "swarm",
                    "Skeleton": "swarm",
                    "Minion": "swarm",
                    "Wizard": "splasher",
                    "Valkyrie": "splasher",
                    "Dragon": "splasher",
                    "Archer": "support",
                    "Musketeer": "support",
                    "Balloon": "tank"
                }
            }
        elif template_type == "phase_based":
            return {
                # Phase definitions
                "early_phase_end": random.randint(30, 60),
                "mid_phase_end": random.randint(90, 150),
                # Phase strategies
                "early_phase_strategy": {
                    "aggression": random.uniform(0.0, 0.5),
                    "elixir_threshold": random.uniform(6, 10),
                    "deploy_distance": random.uniform(10, 20),
                    "troop_preferences": {
                        "cheap": random.uniform(1.0, 3.0),
                        "medium": random.uniform(0.5, 1.5),
                        "expensive": random.uniform(0.2, 0.8)
                    }
                },
                "mid_phase_strategy": {
                    "aggression": random.uniform(0.3, 0.8),
                    "elixir_threshold": random.uniform(4, 8),
                    "deploy_distance": random.uniform(5, 15),
                    "troop_preferences": {
                        "cheap": random.uniform(0.5, 1.5),
                        "medium": random.uniform(1.0, 2.0),
                        "expensive": random.uniform(0.8, 1.5)
                    }
                },
                "late_phase_strategy": {
                    "aggression": random.uniform(0.6, 1.0),
                    "elixir_threshold": random.uniform(2, 6),
                    "deploy_distance": random.uniform(0, 10),
                    "troop_preferences": {
                        "cheap": random.uniform(0.2, 1.0),
                        "medium": random.uniform(0.8, 1.5),
                        "expensive": random.uniform(1.0, 2.0)
                    }
                },
                # Troop categorization
                "troop_costs": {
                    "Archer": "cheap",
                    "Barbarian": "cheap", 
                    "Knight": "cheap",
                    "Minion": "cheap",
                    "Skeleton": "cheap",
                    "Dragon": "medium",
                    "Valkyrie": "medium",
                    "Musketeer": "medium",
                    "Giant": "expensive",
                    "Prince": "expensive",
                    "Wizard": "expensive",
                    "Balloon": "expensive"
                }
            }
        else:
            logger.warning(f"Unknown template type: {template_type}")
            return {}
    
    def _tournament_selection(
        self, 
        population: List[Dict[str, Any]], 
        fitness_scores: Dict[str, float], 
        tournament_size: int
    ) -> Dict[str, Any]:
        """
        Select a strategy using tournament selection.
        
        Args:
            population: List of strategies to select from
            fitness_scores: Fitness scores for each strategy
            tournament_size: Number of strategies to include in each tournament
            
        Returns:
            The selected strategy
        """
        if not population:
            raise ValueError("Cannot perform tournament selection on empty population")
        
        # Select random strategies for the tournament
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Select the best strategy from the tournament
        return max(tournament, key=lambda s: fitness_scores.get(s["id"], 0))
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform crossover between two parent strategies.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Child strategy
        """
        # Create a deep copy of parent1 as the base for the child
        child = copy.deepcopy(parent1)
        
        # If parents are of different template types, just return a copy of the first parent
        if parent1["template_type"] != parent2["template_type"]:
            return child
        
        # Perform parameter-level crossover
        child_params = {}
        template_type = parent1["template_type"]
        
        for key, value in parent1["parameters"].items():
            # For each top-level parameter, randomly choose between parents
            if random.random() < 0.5 and key in parent2["parameters"]:
                child_params[key] = copy.deepcopy(parent2["parameters"][key])
            else:
                child_params[key] = copy.deepcopy(value)
                
            # For dictionary parameters, we can do more granular crossover
            if isinstance(value, dict) and isinstance(parent2["parameters"].get(key, None), dict):
                # For each sub-parameter, randomly choose between parents
                for subkey in value:
                    if random.random() < 0.5 and subkey in parent2["parameters"][key]:
                        child_params[key][subkey] = copy.deepcopy(parent2["parameters"][key][subkey])
        
        child["parameters"] = child_params
        return child
    
    def _mutate(self, strategy: Dict[str, Any], mutation_rate: float) -> None:
        """
        Mutate a strategy in place.
        
        Args:
            strategy: Strategy to mutate
            mutation_rate: Probability of mutation for each parameter
        """
        template_type = strategy["template_type"]
        
        # Get a fresh set of random parameters for this template type
        random_params = self._generate_random_parameters(template_type)
        
        # Mutate top-level parameters
        for key, value in strategy["parameters"].items():
            if random.random() < mutation_rate:
                if key in random_params:
                    strategy["parameters"][key] = random_params[key]
            
            # For dictionary parameters, we can do more granular mutation
            if isinstance(value, dict):
                for subkey in value:
                    if random.random() < mutation_rate:
                        if key in random_params and subkey in random_params[key]:
                            strategy["parameters"][key][subkey] = random_params[key][subkey]
                            
            # For numeric parameters, we can also do small tweaks
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                # Apply a small random change (±20%)
                if isinstance(value, int):
                    delta = max(1, abs(int(value * 0.2)))
                    strategy["parameters"][key] = max(0, value + random.randint(-delta, delta))
                else:  # float
                    delta = abs(value * 0.2)
                    strategy["parameters"][key] = max(0, value + random.uniform(-delta, delta))
    
    def _generate_strategy_file(self, strategy: Dict[str, Any]) -> str:
        """
        Generate a strategy file for a given strategy.
        
        Args:
            strategy: Strategy data
            
        Returns:
            Path to the generated strategy file
        """
        # Create directory for this generation if it doesn't exist
        generation_dir = os.path.join(EVOLVED_STRATEGIES_DIR, f"generation_{strategy['generation']}")
        os.makedirs(generation_dir, exist_ok=True)
        
        # Generate file path
        file_path = os.path.join(generation_dir, f"{strategy['id']}.py")
        
        # For static strategies, just copy the file
        if strategy.get("is_static", False) and "source_file" in strategy:
            if os.path.exists(strategy["source_file"]):
                with open(strategy["source_file"], 'r') as f:
                    content = f.read()
                
                with open(file_path, 'w') as f:
                    f.write(content)
                
                return file_path
        
        # For generated strategies, use the template
        code = self.templates.generate_strategy_code(
            strategy["template_type"], 
            strategy["parameters"],
            strategy["name"]
        )
        
        # Write the code to file
        with open(file_path, 'w') as f:
            f.write(code)
        
        return file_path