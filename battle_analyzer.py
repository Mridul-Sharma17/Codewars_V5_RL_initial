"""
Tower Defense RL - Battle Analyzer

This module analyzes battle results to calculate fitness scores and extract insights.
"""

import math
import logging
from typing import Dict, List, Any, Tuple

from rl_config import *

logger = logging.getLogger('battle_analyzer')

class BattleAnalyzer:
    """Analyzes battle results to calculate fitness scores and extract insights."""
    
    def __init__(self):
        """Initialize the battle analyzer."""
        self.fitness_weights = FITNESS_WEIGHTS
    
    def calculate_fitness(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]],
        generation: int
    ) -> Dict[str, float]:
        """
        Calculate fitness scores for all strategies.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            battle_metrics: Dictionary of battle_id -> battle_data
            generation: Current generation number
            
        Returns:
            Dictionary of strategy_id -> fitness_score
        """
        # Update basic metrics for each strategy
        self._update_basic_metrics(strategies, battle_metrics)
        
        # Calculate Wilson score for each strategy
        self._calculate_wilson_scores(strategies)
        
        # Calculate elite performance (performance against top strategies)
        self._calculate_elite_performance(strategies, battle_metrics)
        
        # Calculate versatility (performance against different strategy archetypes)
        self._calculate_versatility(strategies, battle_metrics)
        
        # Calculate final fitness scores
        fitness_scores = {}
        for strategy_id, strategy in strategies.items():
            # Weighted combination of metrics
            wilson_score = strategy["metrics"]["wilson_score"]
            elite_performance = strategy["metrics"]["elite_performance"]
            avg_tower_health = strategy["metrics"]["avg_tower_health"] / 7032.0  # Normalize by max tower health
            versatility = strategy["metrics"]["versatility"]
            
            fitness_scores[strategy_id] = (
                self.fitness_weights["wilson_score"] * wilson_score +
                self.fitness_weights["elite_performance"] * elite_performance +
                self.fitness_weights["tower_health"] * avg_tower_health +
                self.fitness_weights["versatility"] * versatility
            )
        
        logger.info(f"Calculated fitness scores for {len(fitness_scores)} strategies")
        return fitness_scores
    
    def get_best_strategy(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get the best strategy based on fitness scores.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            battle_metrics: Dictionary of battle_id -> battle_data
            
        Returns:
            The best strategy
        """
        # Calculate fitness scores
        fitness_scores = self.calculate_fitness(strategies, battle_metrics, 0)
        
        # Find the strategy with the highest fitness score
        best_strategy_id = max(fitness_scores, key=fitness_scores.get)
        
        return strategies[best_strategy_id]
    
    def _update_basic_metrics(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Update basic metrics for each strategy.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            battle_metrics: Dictionary of battle_id -> battle_data
        """
        # Reset metrics
        for strategy_id, strategy in strategies.items():
            strategy["metrics"]["games_played"] = 0
            strategy["metrics"]["wins"] = 0
            strategy["metrics"]["losses"] = 0
            strategy["metrics"]["avg_tower_health"] = 0
        
        # Update metrics based on battle results
        for battle_id, battle in battle_metrics.items():
            strategy1_id = battle["strategy1_id"]
            strategy2_id = battle["strategy2_id"]
            winner_id = battle["winner_id"]
            
            if strategy1_id in strategies:
                strategies[strategy1_id]["metrics"]["games_played"] += 1
                
                if winner_id == strategy1_id:
                    strategies[strategy1_id]["metrics"]["wins"] += 1
                elif winner_id is not None:
                    strategies[strategy1_id]["metrics"]["losses"] += 1
                
                if battle["strategy1_health"] is not None:
                    current_avg = strategies[strategy1_id]["metrics"]["avg_tower_health"]
                    games_played = strategies[strategy1_id]["metrics"]["games_played"]
                    if games_played > 1:
                        strategies[strategy1_id]["metrics"]["avg_tower_health"] = (
                            (current_avg * (games_played - 1) + battle["strategy1_health"]) / games_played
                        )
                    else:
                        strategies[strategy1_id]["metrics"]["avg_tower_health"] = battle["strategy1_health"]
            
            if strategy2_id in strategies:
                strategies[strategy2_id]["metrics"]["games_played"] += 1
                
                if winner_id == strategy2_id:
                    strategies[strategy2_id]["metrics"]["wins"] += 1
                elif winner_id is not None:
                    strategies[strategy2_id]["metrics"]["losses"] += 1
                
                if battle["strategy2_health"] is not None:
                    current_avg = strategies[strategy2_id]["metrics"]["avg_tower_health"]
                    games_played = strategies[strategy2_id]["metrics"]["games_played"]
                    if games_played > 1:
                        strategies[strategy2_id]["metrics"]["avg_tower_health"] = (
                            (current_avg * (games_played - 1) + battle["strategy2_health"]) / games_played
                        )
                    else:
                        strategies[strategy2_id]["metrics"]["avg_tower_health"] = battle["strategy2_health"]
        
        # Calculate win rate
        for strategy_id, strategy in strategies.items():
            wins = strategy["metrics"]["wins"]
            games_played = strategy["metrics"]["games_played"]
            strategy["metrics"]["win_rate"] = wins / games_played if games_played > 0 else 0
    
    def _calculate_wilson_scores(self, strategies: Dict[str, Dict[str, Any]]) -> None:
        """
        Calculate Wilson score for each strategy.
        
        The Wilson score is a confidence-adjusted win rate that accounts for sample size.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
        """
        # Calculate Wilson score for each strategy
        for strategy_id, strategy in strategies.items():
            wins = strategy["metrics"]["wins"]
            losses = strategy["metrics"]["losses"]
            total = wins + losses
            
            if total == 0:
                strategy["metrics"]["wilson_score"] = 0
                continue
            
            # Calculate Wilson score with 95% confidence
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
            z = 1.96  # 95% confidence
            p = wins / total
            
            numerator = p + z*z/(2*total) + z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
            denominator = 1 + z*z/total
            
            wilson_score = numerator / denominator
            strategy["metrics"]["wilson_score"] = wilson_score
    
    def _calculate_elite_performance(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Calculate elite performance for each strategy.
        
        Elite performance is the win rate against the top strategies.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            battle_metrics: Dictionary of battle_id -> battle_data
        """
        # Identify top strategies (top 25%)
        sorted_strategies = sorted(
            strategies.values(),
            key=lambda s: s["metrics"]["wilson_score"],
            reverse=True
        )
        
        elite_count = max(2, len(sorted_strategies) // 4)
        elite_ids = {s["id"] for s in sorted_strategies[:elite_count]}
        
        # Calculate elite performance for each strategy
        for strategy_id, strategy in strategies.items():
            elite_games = 0
            elite_wins = 0
            
            # Find battles against elite strategies
            for battle_id, battle in battle_metrics.items():
                # Check if this strategy was in the battle
                is_strategy1 = battle["strategy1_id"] == strategy_id
                is_strategy2 = battle["strategy2_id"] == strategy_id
                
                if not (is_strategy1 or is_strategy2):
                    continue
                
                # Check if the opponent was an elite strategy
                opponent_id = battle["strategy2_id"] if is_strategy1 else battle["strategy1_id"]
                if opponent_id not in elite_ids:
                    continue
                
                # Count the battle
                elite_games += 1
                
                # Check if this strategy won
                if battle["winner_id"] == strategy_id:
                    elite_wins += 1
            
            # Calculate elite performance
            if elite_games > 0:
                strategy["metrics"]["elite_performance"] = elite_wins / elite_games
            else:
                strategy["metrics"]["elite_performance"] = 0
    
    def _calculate_versatility(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Calculate versatility for each strategy.
        
        Versatility is a measure of how well a strategy performs against different archetypes.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            battle_metrics: Dictionary of battle_id -> battle_data
        """
        # Group strategies by template type
        archetypes = {}
        for strategy_id, strategy in strategies.items():
            template_type = strategy["template_type"]
            if template_type not in archetypes:
                archetypes[template_type] = []
            archetypes[template_type].append(strategy_id)
        
        # Calculate versatility for each strategy
        for strategy_id, strategy in strategies.items():
            archetype_performance = {}
            
            # Calculate performance against each archetype
            for archetype, archetype_strategies in archetypes.items():
                archetype_games = 0
                archetype_wins = 0
                
                # Find battles against this archetype
                for battle_id, battle in battle_metrics.items():
                    # Check if this strategy was in the battle
                    is_strategy1 = battle["strategy1_id"] == strategy_id
                    is_strategy2 = battle["strategy2_id"] == strategy_id
                    
                    if not (is_strategy1 or is_strategy2):
                        continue
                    
                    # Check if the opponent was from this archetype
                    opponent_id = battle["strategy2_id"] if is_strategy1 else battle["strategy1_id"]
                    if opponent_id not in archetype_strategies:
                        continue
                    
                    # Count the battle
                    archetype_games += 1
                    
                    # Check if this strategy won
                    if battle["winner_id"] == strategy_id:
                        archetype_wins += 1
                
                # Calculate performance against this archetype
                if archetype_games > 0:
                    archetype_performance[archetype] = archetype_wins / archetype_games
                else:
                    archetype_performance[archetype] = 0
            
            # Calculate overall versatility
            if len(archetype_performance) > 0:
                # Versatility is the harmonic mean of performances against each archetype
                if sum(1 for perf in archetype_performance.values() if perf > 0) > 0:
                    denominator = sum(1 / perf if perf > 0 else 0 for perf in archetype_performance.values())
                    if denominator > 0:
                        strategy["metrics"]["versatility"] = len(archetype_performance) / denominator
                    else:
                        strategy["metrics"]["versatility"] = 0
                else:
                    strategy["metrics"]["versatility"] = 0
            else:
                strategy["metrics"]["versatility"] = 0