"""
Enhanced strategy selection using multi-objective fitness function.
Includes Wilson score confidence intervals, opponent strength weighting,
and performance metrics beyond simple win-loss records.
"""

import math
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Any, Tuple, Optional

class StrategyEvaluator:
    """
    Advanced evaluation system for tower defense strategies.
    Uses multi-objective fitness function to rank strategies.
    """
    
    def __init__(self, confidence=0.95, elite_opponent_bonus=1.5):
        """
        Initialize the strategy evaluator.
        
        Args:
            confidence: Confidence level for Wilson score (0.95 = 95%)
            elite_opponent_bonus: Multiplier for wins against top opponents
        """
        self.confidence = confidence
        self.elite_opponent_bonus = elite_opponent_bonus
        
        # Keep track of strong opponents for bonus scoring
        self.strong_opponents = set()
        # Cache opponent strength ratings
        self.opponent_strength = {}
    
    def calculate_wilson_score(self, wins: int, games: int) -> float:
        """
        Calculate Wilson score lower bound for binomial parameter.
        This gives a more accurate estimate of win rate when few games have been played.
        
        Args:
            wins: Number of wins
            games: Number of games played
            
        Returns:
            Lower bound of Wilson score confidence interval
        """
        if games == 0:
            return 0
        
        # Get z-score for desired confidence
        z = norm.ppf((1 + self.confidence) / 2)
        
        # Calculate Wilson score
        p = float(wins) / games
        denominator = 1 + z**2 / games
        centre_adjusted_probability = p + z**2 / (2 * games)
        adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * games)) / games)
        
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        return lower_bound
    
    def identify_strong_opponents(self, battle_metrics: List[Dict[str, Any]], 
                                  min_games: int = 10, top_percent: float = 0.2) -> None:
        """
        Identify strong opponents based on their performance in battles.
        
        Args:
            battle_metrics: List of battle result dictionaries
            min_games: Minimum games required to be considered
            top_percent: Percentage of top performers to mark as strong
        """
        # Count wins and games for each strategy
        strategy_wins = {}
        strategy_games = {}
        
        for battle in battle_metrics:
            team1 = battle.get("team1_file", "")
            team2 = battle.get("team2_file", "")
            
            if team1:
                strategy_games[team1] = strategy_games.get(team1, 0) + 1
                strategy_wins[team1] = strategy_wins.get(team1, 0) + battle.get("team1_win", 0)
                
            if team2:
                strategy_games[team2] = strategy_games.get(team2, 0) + 1
                strategy_wins[team2] = strategy_wins.get(team2, 0) + battle.get("team2_win", 0)
        
        # Calculate win rates and Wilson scores
        strategy_scores = []
        for strategy_id, games in strategy_games.items():
            if games >= min_games:
                wins = strategy_wins.get(strategy_id, 0)
                win_rate = float(wins) / games if games > 0 else 0
                wilson_score = self.calculate_wilson_score(wins, games)
                
                strategy_scores.append((strategy_id, win_rate, wilson_score, games))
        
        # Sort by Wilson score
        strategy_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Mark top performers as strong
        strong_count = max(1, int(len(strategy_scores) * top_percent))
        self.strong_opponents = set(strategy[0] for strategy in strategy_scores[:strong_count])
        
        # Cache opponent strength ratings (normalized Wilson score)
        if strategy_scores:
            max_score = strategy_scores[0][2]
            min_score = strategy_scores[-1][2] if len(strategy_scores) > 1 else 0
            score_range = max(0.01, max_score - min_score)  # Avoid division by zero
            
            self.opponent_strength = {
                strategy_id: 0.5 + 0.5 * ((wilson - min_score) / score_range)  # Scale to 0.5-1.0
                for strategy_id, _, wilson, _ in strategy_scores
            }
            
            # Ensure Pratyaksh always gets a high rating
            for strategy_id in self.opponent_strength:
                if "pratyaksh" in strategy_id.lower():
                    self.opponent_strength[strategy_id] = max(0.95, self.opponent_strength.get(strategy_id, 0.95))
            
            print(f"Identified {len(self.strong_opponents)} strong opponents")
            for opponent in self.strong_opponents:
                print(f"  - {opponent} (strength: {self.opponent_strength.get(opponent, 0):.2f})")
    
    def calculate_multiobjective_fitness(self, 
                                         strategy_id: str, 
                                         battle_metrics: List[Dict[str, Any]],
                                         min_games: int = 3) -> Dict[str, Any]:
        """
        Calculate a multi-objective fitness score for a strategy.
        
        Args:
            strategy_id: ID of the strategy to evaluate
            battle_metrics: List of all battle results
            min_games: Minimum number of games required for evaluation
            
        Returns:
            Dictionary with fitness metrics and overall score
        """
        # Extract battles for this strategy
        team1_battles = [b for b in battle_metrics if b.get("team1_file", "") == strategy_id]
        team2_battles = [b for b in battle_metrics if b.get("team2_file", "") == strategy_id]
        
        total_games = len(team1_battles) + len(team2_battles)
        if total_games < min_games:
            return {
                "strategy_id": strategy_id,
                "games": total_games,
                "wins": 0,
                "fitness": 0,
                "wilson_score": 0,
                "strong_opponent_score": 0,
                "health_preservation": 0,
                "elixir_efficiency": 0
            }
        
        # Calculate wins and losses
        wins = sum(b.get("team1_win", 0) for b in team1_battles)
        wins += sum(b.get("team2_win", 0) for b in team2_battles)
        
        # Basic win rate statistics
        win_rate = wins / total_games if total_games > 0 else 0
        wilson_score = self.calculate_wilson_score(wins, total_games)
        
        # Performance against strong opponents
        strong_opponent_games = 0
        strong_opponent_wins = 0
        
        for battle in team1_battles:
            opponent = battle.get("team2_file", "")
            if opponent in self.strong_opponents:
                strong_opponent_games += 1
                strong_opponent_wins += battle.get("team1_win", 0)
                
        for battle in team2_battles:
            opponent = battle.get("team1_file", "")
            if opponent in self.strong_opponents:
                strong_opponent_games += 1
                strong_opponent_wins += battle.get("team2_win", 0)
                
        strong_opponent_score = 0
        if strong_opponent_games > 0:
            strong_opponent_score = self.calculate_wilson_score(
                strong_opponent_wins, strong_opponent_games
            )
        
        # Weighted opponent strength score
        weighted_wins = 0
        weighted_games = 0
        for battle in team1_battles:
            opponent = battle.get("team2_file", "")
            opponent_weight = self.opponent_strength.get(opponent, 0.5)
            weighted_games += opponent_weight
            if battle.get("team1_win", 0) == 1:
                weighted_wins += opponent_weight
                
        for battle in team2_battles:
            opponent = battle.get("team1_file", "")
            opponent_weight = self.opponent_strength.get(opponent, 0.5)
            weighted_games += opponent_weight
            if battle.get("team2_win", 0) == 1:
                weighted_wins += opponent_weight
                
        weighted_win_rate = weighted_wins / weighted_games if weighted_games > 0 else 0
        
        # Tower health preservation
        own_health_total = 0
        opponent_health_diff_total = 0
        games_with_health = 0
        
        for battle in team1_battles:
            if "team1_health" in battle and "team2_health" in battle:
                max_health = 10000  # Assumption
                own_health_total += battle["team1_health"] / max_health
                opponent_health_diff_total += ((battle["team1_health"] - battle["team2_health"]) / max_health)
                games_with_health += 1
                
        for battle in team2_battles:
            if "team1_health" in battle and "team2_health" in battle:
                max_health = 10000  # Assumption
                own_health_total += battle["team2_health"] / max_health
                opponent_health_diff_total += ((battle["team2_health"] - battle["team1_health"]) / max_health)
                games_with_health += 1
                
        health_preservation = own_health_total / games_with_health if games_with_health > 0 else 0
        health_diff = opponent_health_diff_total / games_with_health if games_with_health > 0 else 0
        
        # Elixir efficiency (if available in metrics)
        elixir_efficiency = 0  # Default
        
        # Calculate final multi-objective fitness
        # Weights for different components
        w1 = 0.60  # Wilson score (baseline statistical win rate)
        w2 = 0.15  # Performance against strong opponents
        w3 = 0.15  # Health preservation
        w4 = 0.10  # Health difference
        
        fitness = (
            w1 * wilson_score + 
            w2 * (strong_opponent_score if strong_opponent_games > 0 else weighted_win_rate) +
            w3 * health_preservation +
            w4 * (max(0, health_diff) + 1) / 2  # Normalize to 0-1 range
        )
        
        # If this is Pratyaksh or another benchmark, we don't want to select it for evolution
        is_benchmark = "pratyaksh" in strategy_id.lower() or strategy_id in ["baseline", "aggressive", "defensive", "balanced"]
        if is_benchmark:
            fitness *= 0.5  # Discourage selection of benchmark strategies
        
        return {
            "strategy_id": strategy_id,
            "games": total_games,
            "wins": wins,
            "win_rate": win_rate,
            "wilson_score": wilson_score,
            "strong_opponent_score": strong_opponent_score,
            "health_preservation": health_preservation,
            "health_diff": health_diff,
            "weighted_win_rate": weighted_win_rate,
            "strong_opponent_games": strong_opponent_games, 
            "fitness": fitness,
            "is_benchmark": is_benchmark
        }
    
    def select_best_strategies(self, battle_metrics: List[Dict[str, Any]], 
                               min_games: int = 3, league: str = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Select best strategies based on multi-objective fitness.
        Can filter by specific league (bronze, silver, gold, master) if provided.
        
        Args:
            battle_metrics: List of battle result dictionaries
            min_games: Minimum number of games required for consideration
            league: Optional league filter (None returns all strategies)
            
        Returns:
            List of tuples (strategy_id, stats_dict) sorted by fitness
        """
        # First identify strong opponents to calculate opponent strength
        self.identify_strong_opponents(battle_metrics)
        
        # Get all strategy IDs from battle metrics
        strategy_ids = set()
        for battle in battle_metrics:
            team1 = battle.get("team1_file", "")
            team2 = battle.get("team2_file", "")
            if team1:
                strategy_ids.add(team1)
            if team2:
                strategy_ids.add(team2)
        
        # Calculate fitness for each strategy
        strategy_fitness = []
        for strategy_id in strategy_ids:
            fitness_data = self.calculate_multiobjective_fitness(
                strategy_id, battle_metrics, min_games
            )
            
            # Only include strategies with enough games
            if fitness_data["games"] >= min_games:
                strategy_fitness.append((strategy_id, fitness_data))
        
        # Sort by fitness
        sorted_strategies = sorted(
            strategy_fitness,
            key=lambda x: x[1]["fitness"],
            reverse=True
        )
        
        # Apply league filtering if requested
        if league:
            # Split into leagues by fitness
            total = len(sorted_strategies)
            if total == 0:
                return []
                
            if league == "bronze":
                # Bottom 40%
                cutoff = max(0, int(total * 0.6))
                return sorted_strategies[cutoff:]
            elif league == "silver":
                # Middle 30% 
                start = max(0, int(total * 0.3))
                end = max(0, int(total * 0.6))
                return sorted_strategies[start:end]
            elif league == "gold":
                # Top 20-30%
                start = max(0, int(total * 0.1))
                end = max(0, int(total * 0.3))
                return sorted_strategies[start:end]
            elif league == "master":
                # Top 10%
                end = max(0, int(total * 0.1))
                return sorted_strategies[:end]
        
        return sorted_strategies

    def get_league_assignment(self, strategy_id: str, battle_metrics: List[Dict[str, Any]], 
                              min_games: int = 3) -> str:
        """
        Determine which league a strategy belongs to.
        
        Args:
            strategy_id: Strategy to evaluate
            battle_metrics: List of battle results
            min_games: Minimum games required for league assignment
            
        Returns:
            League name: "master", "gold", "silver", "bronze" or "unranked"
        """
        # First, get all strategies sorted by fitness
        sorted_strategies = self.select_best_strategies(battle_metrics, min_games)
        
        # Find the index of our strategy
        strategy_index = -1
        for i, (sid, _) in enumerate(sorted_strategies):
            if sid == strategy_id:
                strategy_index = i
                break
                
        if strategy_index < 0:
            return "unranked"
            
        # Determine league based on position
        total = len(sorted_strategies)
        if total == 0:
            return "unranked"
            
        # Calculate percentile
        percentile = strategy_index / total
        
        if percentile < 0.1:
            return "master"
        elif percentile < 0.3:
            return "gold"
        elif percentile < 0.6:
            return "silver"
        else:
            return "bronze"
    
    def select_opponents(self, strategy_id: str, 
                         battle_metrics: List[Dict[str, Any]],
                         all_strategy_ids: List[str],
                         battles_needed: int,
                         min_games_for_league: int = 3,
                         strong_opponent_ratio: float = 0.3) -> List[str]:
        """
        Select appropriate opponents for a strategy based on its league.
        Ensures a mix of similarly skilled and stronger opponents.
        
        Args:
            strategy_id: Strategy needing opponents
            battle_metrics: List of all battle results
            all_strategy_ids: List of all available strategy IDs
            battles_needed: Number of opponents to select
            min_games_for_league: Minimum games to assign a league
            strong_opponent_ratio: Ratio of battles against strong opponents
            
        Returns:
            List of selected opponent strategy IDs
        """
        # Get league assignment for this strategy
        league = self.get_league_assignment(strategy_id, battle_metrics, min_games_for_league)
        
        # If unranked, battle against a mix of strategies
        if league == "unranked":
            # Mix of all strategies available
            opponents = random.sample(all_strategy_ids, min(len(all_strategy_ids), battles_needed))
            return opponents
            
        # Determine how many strong opponents to face
        strong_count = max(1, int(battles_needed * strong_opponent_ratio))
        similar_count = battles_needed - strong_count
        
        # Get strategies by league
        master_leagues = [s for s, _ in self.select_best_strategies(battle_metrics, min_games_for_league, "master")]
        gold_leagues = [s for s, _ in self.select_best_strategies(battle_metrics, min_games_for_league, "gold")]
        silver_leagues = [s for s, _ in self.select_best_strategies(battle_metrics, min_games_for_league, "silver")]
        bronze_leagues = [s for s, _ in self.select_best_strategies(battle_metrics, min_games_for_league, "bronze")]
        
        # Select opponent pools based on current league
        strong_pool = []
        similar_pool = []
        
        if league == "master":
            strong_pool = master_leagues
            similar_pool = master_leagues + gold_leagues[:len(gold_leagues)//2]
        elif league == "gold":
            strong_pool = master_leagues + gold_leagues[:len(gold_leagues)//2]
            similar_pool = gold_leagues + silver_leagues[:len(silver_leagues)//2]
        elif league == "silver":
            strong_pool = gold_leagues + silver_leagues[:len(silver_leagues)//2]
            similar_pool = silver_leagues + bronze_leagues[:len(bronze_leagues)//2]
        else:  # bronze
            strong_pool = silver_leagues + bronze_leagues[:len(bronze_leagues)//2]
            similar_pool = bronze_leagues
            
        # Remove self from pools
        if strategy_id in strong_pool:
            strong_pool.remove(strategy_id)
        if strategy_id in similar_pool:
            similar_pool.remove(strategy_id)
            
        # Ensure Pratyaksh and other key opponents are in the strong pool
        for s in all_strategy_ids:
            if "pratyaksh" in s.lower() and s not in strong_pool:
                strong_pool.append(s)
                
        # Add fallback in case pools are too small
        if not strong_pool:
            strong_pool = all_strategy_ids.copy()
            if strategy_id in strong_pool:
                strong_pool.remove(strategy_id)
                
        if not similar_pool:
            similar_pool = all_strategy_ids.copy()
            if strategy_id in similar_pool:
                similar_pool.remove(strategy_id)
        
        # Select opponents
        selected_opponents = []
        
        # First, add strong opponents
        if strong_pool:
            # Sample with replacement if there aren't enough strong opponents
            if len(strong_pool) < strong_count:
                # Select all available strong opponents
                selected_opponents.extend(strong_pool)
                # Fill the rest with random selection from the pool
                for _ in range(strong_count - len(strong_pool)):
                    selected_opponents.append(random.choice(strong_pool))
            else:
                # Select without replacement
                selected_opponents.extend(random.sample(strong_pool, strong_count))
        
        # Then add opponents of similar skill
        if similar_pool:
            # Sample with replacement if needed
            if len(similar_pool) < similar_count:
                # Select all available similar opponents
                for opponent in similar_pool:
                    if opponent not in selected_opponents:
                        selected_opponents.append(opponent)
                # Fill the rest with random selection
                for _ in range(similar_count - (len(similar_pool) - len(set(similar_pool) & set(selected_opponents)))):
                    opponent = random.choice(similar_pool)
                    selected_opponents.append(opponent)
            else:
                # Select without replacement
                additional_opponents = []
                remaining_pool = [s for s in similar_pool if s not in selected_opponents]
                if remaining_pool:
                    additional_opponents = random.sample(
                        remaining_pool,
                        min(similar_count, len(remaining_pool))
                    )
                # If we still need more, pick randomly from the full pool
                while len(additional_opponents) < similar_count:
                    opponent = random.choice(similar_pool)
                    if len(additional_opponents) < similar_count:
                        additional_opponents.append(opponent)
                
                selected_opponents.extend(additional_opponents)
        
        return selected_opponents


# For backward compatibility
def calculate_wilson_score(wins, games, confidence=0.95):
    """Legacy function to calculate Wilson score confidence interval."""
    evaluator = StrategyEvaluator(confidence=confidence)
    return evaluator.calculate_wilson_score(wins, games)

def select_best_strategies(battle_metrics, min_games=3, confidence=0.95):
    """Legacy function to select best strategies based on Wilson score."""
    evaluator = StrategyEvaluator(confidence=confidence)
    return [(s, {"wins": d["wins"], 
                "games": d["games"], 
                "win_rate": d["win_rate"], 
                "wilson_score": d["wilson_score"]})
            for s, d in evaluator.select_best_strategies(battle_metrics, min_games)]


# Add imports for league-based opponent selection
import random