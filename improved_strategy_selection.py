"""
Improved strategy selection using Wilson score confidence intervals.
This provides a more statistically sound way to rank strategies, especially with few games.
"""

import math
import numpy as np
from scipy.stats import norm

def calculate_wilson_score(wins, games, confidence=0.95):
    """
    Calculate Wilson score lower bound for binomial parameter.
    This gives a more accurate estimate of win rate when few games have been played.
    
    Args:
        wins: Number of wins
        games: Number of games played
        confidence: Confidence level (0.95 = 95%)
        
    Returns:
        Lower bound of Wilson score confidence interval
    """
    if games == 0:
        return 0
    
    # Get z-score for desired confidence
    z = norm.ppf((1 + confidence) / 2)
    
    # Calculate Wilson score
    p = float(wins) / games
    denominator = 1 + z**2 / games
    centre_adjusted_probability = p + z**2 / (2 * games)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * games)) / games)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    return lower_bound

def select_best_strategies(battle_metrics, min_games=3, confidence=0.95):
    """
    Select best strategies based on Wilson score confidence interval.
    This accounts for both win rate and number of games played.
    
    Args:
        battle_metrics: List of battle result dictionaries
        min_games: Minimum number of games required for consideration
        confidence: Confidence level for Wilson score (0.95 = 95%)
        
    Returns:
        List of tuples (strategy_id, stats_dict) sorted by Wilson score
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
    
    # Calculate Wilson scores
    strategy_stats = {}
    for strategy_id, games in strategy_games.items():
        if games >= min_games:
            wins = strategy_wins.get(strategy_id, 0)
            win_rate = float(wins) / games if games > 0 else 0
            wilson_score = calculate_wilson_score(wins, games, confidence)
            
            strategy_stats[strategy_id] = {
                "wins": wins,
                "games": games,
                "win_rate": win_rate,
                "wilson_score": wilson_score
            }
    
    # Sort by Wilson score
    sorted_strategies = sorted(
        strategy_stats.items(), 
        key=lambda x: x[1]["wilson_score"], 
        reverse=True
    )
    
    return sorted_strategies