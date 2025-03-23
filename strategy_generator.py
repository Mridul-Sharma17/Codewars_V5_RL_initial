"""
Tower Defense RL - Strategy Generator

This module handles generating and evolving tower defense strategies.
"""

import os
import random
import uuid
import json
import logging
import copy
import time
from typing import Dict, List, Any

from rl_config import *
from adaptive_templates import AdaptiveTemplates

logger = logging.getLogger('strategy_generator')

class StrategyGenerator:
    """Generates and evolves strategies for tower defense."""
    
    def __init__(self):
        """Initialize the strategy generator."""
        self.templates = AdaptiveTemplates()
        
    def initialize_population(self, size: int, generation: int = 0) -> Dict[str, Dict[str, Any]]:
        """
        Initialize a population of strategies.
        
        Args:
            size: Number of strategies to generate
            generation: Current generation number
            
        Returns:
            Dictionary of strategy_id -> strategy_data
        """
        # Add any static baseline strategies
        strategies = self._init_baseline_strategies()
        
        # Calculate how many new strategies we need to generate
        remaining = size - len(strategies)
        
        if remaining > 0:
            # Generate new strategies
            template_types = self._select_template_types(remaining)
            
            for template_type in template_types:
                # Generate a new strategy
                strategy_id = self._generate_strategy_id(template_type)
                
                # Generate random parameters for this strategy
                parameters = self.templates.generate_parameters(template_type)
                
                # Generate a name for this strategy
                name = self._generate_strategy_name(template_type)
                
                # Create the strategy file
                source_file = self._create_strategy_file(
                    template_type, parameters, strategy_id, name, generation
                )
                
                # Add the strategy to our collection
                strategies[strategy_id] = {
                    "id": strategy_id,
                    "name": name,
                    "template_type": template_type,
                    "parameters": parameters,
                    "source_file": source_file,
                    "generation": generation,
                    "parent_ids": [],
                    "is_static": False,
                    "metrics": self._init_metrics()  # Initialize metrics
                }
            
            logger.info(f"Generated {len(template_types)} initial strategies")
        
        return strategies
    
    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize metrics for a new strategy."""
        return {
            "wins": 0,
            "losses": 0,
            "games_played": 0,
            "win_rate": 0.0,
            "avg_tower_health": 0.0,
            "fitness": 0.0
        }
    
    def _init_baseline_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize baseline strategies."""
        strategies = {}
        
        for baseline in BASELINE_STRATEGIES:
            strategies[baseline["id"]] = copy.deepcopy(baseline)
            strategies[baseline["id"]]["metrics"] = self._init_metrics()
        
        return strategies
    
    def _select_template_types(self, count: int) -> List[str]:
        """
        Select template types for new strategies based on weights.
        
        Args:
            count: Number of template types to select
            
        Returns:
            List of template types
        """
        template_types = []
        
        # Weighted random selection of template types
        templates = list(TEMPLATE_WEIGHTS.keys())
        weights = list(TEMPLATE_WEIGHTS.values())
        
        for _ in range(count):
            template_type = random.choices(templates, weights=weights, k=1)[0]
            template_types.append(template_type)
        
        return template_types
    
    def _generate_strategy_id(self, template_type: str) -> str:
        """
        Generate a unique ID for a strategy.
        
        Args:
            template_type: Type of the strategy template
            
        Returns:
            Unique strategy ID
        """
        # Use first 8 characters of a UUID
        unique_id = uuid.uuid4().hex[:8]
        return f"{template_type}_{unique_id}"
    
    def _generate_strategy_name(self, template_type: str) -> str:
        """
        Generate a human-readable name for a strategy.
        
        Args:
            template_type: Type of the strategy template
            
        Returns:
            Strategy name
        """
        # Convert snake_case to Title Case
        template_name = ' '.join(word.capitalize() for word in template_type.split('_'))
        
        # Add a short random suffix (4 chars)
        suffix = uuid.uuid4().hex[:4]
        
        return f"{template_name} {suffix}"
    
    def _create_strategy_file(self, template_type: str, parameters: Dict[str, Any], 
                             strategy_id: str, name: str, generation: int) -> str:
        """
        Create a strategy file from a template.
        
        Args:
            template_type: Type of the strategy template
            parameters: Strategy parameters
            strategy_id: Unique strategy ID
            name: Strategy name
            generation: Current generation number
            
        Returns:
            Path to the created strategy file
        """
        # Generate the strategy code
        code = self.templates.generate_strategy_code(template_type, parameters, name)
        
        # Create the directory for this generation
        generation_dir = os.path.join(EVOLVED_STRATEGIES_DIR, f"generation_{generation}")
        os.makedirs(generation_dir, exist_ok=True)
        
        # Create the strategy file
        strategy_file = os.path.join(generation_dir, f"{strategy_id}.py")
        with open(strategy_file, 'w') as f:
            f.write(code)
        
        return strategy_file
    
    def evolve_population(self, strategies: Dict[str, Dict[str, Any]], 
                         elite_ids: List[str], generation: int, 
                         population_size: int, mutation_rate: float) -> Dict[str, Dict[str, Any]]:
        """
        Evolve a population of strategies.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            elite_ids: IDs of elite strategies to keep
            generation: Current generation number
            population_size: Target population size
            mutation_rate: Probability of mutation
            
        Returns:
            Dictionary of strategy_id -> strategy_data for the new generation
        """
        # Keep any static baseline strategies
        new_strategies = {}
        for s_id, strategy in strategies.items():
            if strategy.get("is_static", False):
                # Reset metrics for the new generation
                strategy["metrics"] = self._init_metrics()
                new_strategies[s_id] = strategy
        
        # Add elite strategies (without modification)
        for elite_id in elite_ids:
            if elite_id in strategies and not strategies[elite_id].get("is_static", False):
                elite = copy.deepcopy(strategies[elite_id])
                # Reset metrics for the new generation
                elite["metrics"] = self._init_metrics()
                elite["generation"] = generation
                new_strategies[elite_id] = elite
        
        # Calculate how many new strategies we need to generate
        remaining = population_size - len(new_strategies)
        
        if remaining > 0:
            # Select parent strategies for breeding
            parent_ids = self._select_parents(strategies, remaining, elite_ids)
            
            # Generate new strategies through crossover and mutation
            for i in range(remaining):
                parent1_id, parent2_id = parent_ids[i]
                
                # Get the parent strategies
                parent1 = strategies[parent1_id]
                parent2 = strategies[parent2_id]
                
                # Create a child strategy through crossover
                child = self._crossover(parent1, parent2)
                
                # Potentially mutate the child
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                # Generate a new ID and update metadata
                template_type = child["template_type"]
                strategy_id = self._generate_strategy_id(template_type)
                name = self._generate_strategy_name(template_type)
                
                # Create the strategy file
                source_file = self._create_strategy_file(
                    template_type, child["parameters"], strategy_id, name, generation
                )
                
                # Add the child to the new population
                new_strategies[strategy_id] = {
                    "id": strategy_id,
                    "name": name,
                    "template_type": template_type,
                    "parameters": child["parameters"],
                    "source_file": source_file,
                    "generation": generation,
                    "parent_ids": [parent1_id, parent2_id],
                    "is_static": False,
                    "metrics": self._init_metrics()  # Initialize metrics
                }
            
            logger.info(f"Generated {remaining} new strategies through evolution")
        
        return new_strategies
    
    def _select_parents(self, strategies: Dict[str, Dict[str, Any]], 
                       count: int, elite_ids: List[str]) -> List[tuple]:
        """
        Select parents for breeding using tournament selection.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            count: Number of parent pairs to select
            elite_ids: IDs of elite strategies
            
        Returns:
            List of (parent1_id, parent2_id) pairs
        """
        # Get eligible parents (exclude static strategies)
        eligible_ids = [sid for sid, s in strategies.items() if not s.get("is_static", False)]
        
        if len(eligible_ids) < 2:
            # Not enough eligible parents, use elite IDs
            return [(elite_ids[0], elite_ids[0])] * count
        
        # Perform tournament selection
        parent_pairs = []
        
        for _ in range(count):
            # Select first parent through tournament selection
            parent1_id = self._tournament_select(strategies, eligible_ids)
            
            # Select second parent (must be different from first)
            parent2_candidates = [pid for pid in eligible_ids if pid != parent1_id]
            parent2_id = self._tournament_select(strategies, parent2_candidates)
            
            parent_pairs.append((parent1_id, parent2_id))
        
        return parent_pairs
    
    def _tournament_select(self, strategies: Dict[str, Dict[str, Any]], 
                         candidate_ids: List[str]) -> str:
        """
        Select a strategy using tournament selection.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            candidate_ids: IDs of candidate strategies
            
        Returns:
            ID of the selected strategy
        """
        # If no candidates, return None
        if not candidate_ids:
            return None
        
        # Select TOURNAMENT_SIZE random candidates
        tournament_size = min(TOURNAMENT_SIZE, len(candidate_ids))
        tournament = random.sample(candidate_ids, tournament_size)
        
        # Select the candidate with the highest fitness
        best_id = tournament[0]
        best_fitness = strategies[best_id]["metrics"]["fitness"]
        
        for candidate_id in tournament[1:]:
            fitness = strategies[candidate_id]["metrics"]["fitness"]
            if fitness > best_fitness:
                best_id = candidate_id
                best_fitness = fitness
        
        return best_id
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a child strategy through crossover of two parents.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Child strategy
        """
        # Crossover the parameters using template-specific logic
        template_type = parent1["template_type"]
        if template_type != parent2["template_type"]:
            # If different template types, inherit from one parent
            if random.random() < 0.5:
                return {
                    "template_type": parent1["template_type"],
                    "parameters": copy.deepcopy(parent1["parameters"])
                }
            else:
                return {
                    "template_type": parent2["template_type"],
                    "parameters": copy.deepcopy(parent2["parameters"])
                }
        
        # Same template type, perform parameter crossover
        child_params = self.templates.crossover_parameters(
            template_type,
            parent1["parameters"],
            parent2["parameters"]
        )
        
        return {
            "template_type": template_type,
            "parameters": child_params
        }
    
    def _mutate(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate a strategy.
        
        Args:
            strategy: Strategy to mutate
            
        Returns:
            Mutated strategy
        """
        # Either mutate parameters or change template type
        if random.random() < 0.9:  # 90% chance to mutate parameters
            # Mutate the parameters using template-specific logic
            mutated_params = self.templates.mutate_parameters(
                strategy["template_type"],
                strategy["parameters"]
            )
            
            strategy["parameters"] = mutated_params
        else:  # 10% chance to change template type
            # Select a new template type
            templates = list(TEMPLATE_WEIGHTS.keys())
            weights = list(TEMPLATE_WEIGHTS.values())
            new_template_type = random.choices(templates, weights=weights, k=1)[0]
            
            # Generate new parameters for the new template type
            if new_template_type != strategy["template_type"]:
                strategy["template_type"] = new_template_type
                strategy["parameters"] = self.templates.generate_parameters(new_template_type)
        
        return strategy