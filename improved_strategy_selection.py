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
    strategy_positions = {"team1": {}, "team2": {}}
    
    for battle in battle_metrics:
        team1_file = battle.get("team1_file", "")
        team2_file = battle.get("team2_file", "")
        team1_id = get_strategy_id(team1_file)
        team2_id = get_strategy_id(team2_file)
        
        # Skip if we can't identify the strategy IDs
        if not team1_id or not team2_id:
            continue
        
        # Track games played in each position
        strategy_positions["team1"][team1_id] = strategy_positions["team1"].get(team1_id, 0) + 1
        strategy_positions["team2"][team2_id] = strategy_positions["team2"].get(team2_id, 0) + 1
        
        # Update games played
        strategy_games[team1_id] = strategy_games.get(team1_id, 0) + 1
        strategy_games[team2_id] = strategy_games.get(team2_id, 0) + 1
        
        # Update wins based on winner
        winner = battle.get("winner")
        if winner == battle.get("team1_name"):
            # Team 1 won
            strategy_wins[team1_id] = strategy_wins.get(team1_id, 0) + 1
        elif winner == battle.get("team2_name"):
            # Team 2 won
            strategy_wins[team2_id] = strategy_wins.get(team2_id, 0) + 1
    
    # Calculate Wilson scores and other stats
    strategy_stats = {}
    for strategy_id, games in strategy_games.items():
        if games >= min_games:
            wins = strategy_wins.get(strategy_id, 0)
            win_rate = float(wins) / games if games > 0 else 0
            wilson_score = calculate_wilson_score(wins, games, confidence)
            
            # Track position balance (how often played as team1 vs team2)
            team1_count = strategy_positions["team1"].get(strategy_id, 0)
            team2_count = strategy_positions["team2"].get(strategy_id, 0)
            position_balance = abs(team1_count - team2_count) / games if games > 0 else 1.0
            
            strategy_stats[strategy_id] = {
                "wins": wins,
                "games": games,
                "win_rate": win_rate,
                "wilson_score": wilson_score,
                "team1_games": team1_count,
                "team2_games": team2_count,
                "position_balance": position_balance  # Lower is better (more balanced)
            }
    
    # Sort by Wilson score
    sorted_strategies = sorted(
        strategy_stats.items(), 
        key=lambda x: x[1]["wilson_score"], 
        reverse=True
    )
    
    return sorted_strategies

def get_strategy_id(filename):
    """Extract strategy ID from filename"""
    if not filename:
        return None
        
    # Remove file extension if present
    if filename.endswith('.py'):
        filename = filename[:-3]
        
    # Return the filename as the strategy ID
    return filename