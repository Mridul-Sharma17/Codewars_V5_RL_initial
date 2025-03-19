import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

class BattleAnalyzer:
    """
    Analyzes results from tower defense battles to extract insights and metrics
    for reinforcement learning.
    """
    
    def __init__(self, metrics_file="training_data/battle_metrics.json", 
                 visualization_dir="training_data/visualizations"):
        """
        Initialize the battle analyzer.
        
        Args:
            metrics_file: Path to save battle metrics
            visualization_dir: Directory to save visualizations
        """
        self.metrics_file = metrics_file
        self.visualization_dir = visualization_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Load existing metrics or initialize empty
        self.metrics = self.load_metrics()
        
        # Statistics across all battles
        self.strategy_stats = {}
        self.update_strategy_stats()
    
    def load_metrics(self) -> List[Dict[str, Any]]:
        """
        Load existing battle metrics from file.
        
        Returns:
            List of battle metric dictionaries
        """
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Error loading metrics file. Starting with empty metrics.")
                return []
        else:
            print(f"No existing metrics file found. Starting with empty metrics.")
            return []
    
    def save_metrics(self):
        """Save battle metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved {len(self.metrics)} battle metrics to {self.metrics_file}")
    
    def add_battle_result(self, result: Dict[str, Any]):
        """
        Add a new battle result and analyze it.
        
        Args:
            result: Dictionary containing battle results
        """
        # Add timestamp if not present
        if "timestamp" not in result:
            result["timestamp"] = datetime.now().isoformat()
            
        # Add derived metrics
        result = self._enrich_battle_result(result)
        
        # Add to metrics list
        self.metrics.append(result)
        
        # Update cumulative statistics
        self.update_strategy_stats()
        
        # Save updated metrics
        self.save_metrics()
    
    def _enrich_battle_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add derived metrics to a battle result.
        
        Args:
            result: Battle result dictionary
            
        Returns:
            Enriched battle result with additional metrics
        """
        enriched = result.copy()
        
        # Calculate tower health percentage
        if "team1_health" in result and "team2_health" in result:
            # Assume max tower health is 10000
            max_health = 10000
            enriched["team1_health_pct"] = result["team1_health"] / max_health
            enriched["team2_health_pct"] = result["team2_health"] / max_health
            
            # Calculate health difference
            enriched["health_diff"] = result["team1_health"] - result["team2_health"]
            enriched["health_diff_pct"] = enriched["team1_health_pct"] - enriched["team2_health_pct"]
        
        # Calculate normalized game duration
        if "game_duration" in result:
            # Normalize by total possible duration (1830 frames)
            max_duration = 1830
            enriched["game_duration_pct"] = min(1.0, result["game_duration"] / max_duration)
            
            # Quick win detection
            enriched["quick_win"] = (
                result["game_duration"] < max_duration * 0.7 and 
                "winner" in result and 
                result["winner"] is not None
            )
        
        # Add clean strategy IDs if we can extract them from file names
        if "team1_file" in result:
            team1_id = self._extract_strategy_id(result["team1_file"])
            if team1_id:
                enriched["team1_strategy_id"] = team1_id
                
        if "team2_file" in result:
            team2_id = self._extract_strategy_id(result["team2_file"])
            if team2_id:
                enriched["team2_strategy_id"] = team2_id
        
        return enriched
    
    def _extract_strategy_id(self, filename: str) -> Optional[str]:
        """
        Extract strategy ID from filename.
        
        Args:
            filename: Name of strategy file
            
        Returns:
            Strategy ID if found, None otherwise
        """
        # Expected format: strategy_id_*.py
        parts = filename.split('_')
        if len(parts) >= 2:
            if len(parts[0]) <= 3:  # Short prefix like "gen" or "div"
                return f"{parts[0]}_{parts[1]}"
            else:
                return parts[0]
        return None
    
    def update_strategy_stats(self):
        """Update cumulative statistics for each strategy."""
        # Reset stats dictionary
        self.strategy_stats = {}
        
        # Collect all strategy IDs
        strategy_ids = set()
        for battle in self.metrics:
            if "team1_strategy_id" in battle:
                strategy_ids.add(battle["team1_strategy_id"])
            if "team2_strategy_id" in battle:
                strategy_ids.add(battle["team2_strategy_id"])
        
        # Initialize stats for each strategy
        for strategy_id in strategy_ids:
            self.strategy_stats[strategy_id] = {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "avg_tower_health": 0,
                "avg_opponent_health": 0,
                "quick_wins": 0,
                "win_rate": 0,
                "avg_game_duration": 0
            }
        
        # Aggregate metrics for each strategy
        for battle in self.metrics:
            # Skip battles without strategy IDs
            if "team1_strategy_id" not in battle or "team2_strategy_id" not in battle:
                continue
                
            team1_id = battle["team1_strategy_id"]
            team2_id = battle["team2_strategy_id"]
            
            # Update team1 stats
            stats1 = self.strategy_stats[team1_id]
            stats1["games_played"] += 1
            
            if "winner" in battle:
                if battle["winner"] == battle.get("team1_name"):
                    stats1["wins"] += 1
                elif battle["winner"] == battle.get("team2_name"):
                    stats1["losses"] += 1
                elif battle["winner"] == "Tie":
                    stats1["draws"] += 1
            
            # Update health metrics
            if "team1_health" in battle:
                # Running average: new_avg = old_avg * (n-1)/n + new_value/n
                n = stats1["games_played"]
                old_avg = stats1["avg_tower_health"]
                new_value = battle["team1_health"]
                stats1["avg_tower_health"] = old_avg * (n-1)/n + new_value/n
                
            if "team2_health" in battle:
                n = stats1["games_played"]
                old_avg = stats1["avg_opponent_health"]
                new_value = battle["team2_health"]
                stats1["avg_opponent_health"] = old_avg * (n-1)/n + new_value/n
                
            # Update quick win stat
            if battle.get("quick_win", False) and battle.get("winner") == battle.get("team1_name"):
                stats1["quick_wins"] += 1
                
            # Update game duration
            if "game_duration" in battle:
                n = stats1["games_played"]
                old_avg = stats1["avg_game_duration"]
                new_value = battle["game_duration"]
                stats1["avg_game_duration"] = old_avg * (n-1)/n + new_value/n
            
            # Calculate win rate
            if stats1["games_played"] > 0:
                stats1["win_rate"] = stats1["wins"] / stats1["games_played"]
            
            # Same updates for team2
            stats2 = self.strategy_stats[team2_id]
            stats2["games_played"] += 1
            
            if "winner" in battle:
                if battle["winner"] == battle.get("team2_name"):
                    stats2["wins"] += 1
                elif battle["winner"] == battle.get("team1_name"):
                    stats2["losses"] += 1
                elif battle["winner"] == "Tie":
                    stats2["draws"] += 1
            
            if "team2_health" in battle:
                n = stats2["games_played"]
                old_avg = stats2["avg_tower_health"]
                new_value = battle["team2_health"]
                stats2["avg_tower_health"] = old_avg * (n-1)/n + new_value/n
                
            if "team1_health" in battle:
                n = stats2["games_played"]
                old_avg = stats2["avg_opponent_health"]
                new_value = battle["team1_health"]
                stats2["avg_opponent_health"] = old_avg * (n-1)/n + new_value/n
                
            if battle.get("quick_win", False) and battle.get("winner") == battle.get("team2_name"):
                stats2["quick_wins"] += 1
                
            if "game_duration" in battle:
                n = stats2["games_played"]
                old_avg = stats2["avg_game_duration"]
                new_value = battle["game_duration"]
                stats2["avg_game_duration"] = old_avg * (n-1)/n + new_value/n
            
            if stats2["games_played"] > 0:
                stats2["win_rate"] = stats2["wins"] / stats2["games_played"]
    
    def get_top_strategies(self, min_games=3, limit=10) -> List[Dict[str, Any]]:
        """
        Get the top performing strategies sorted by win rate.
        
        Args:
            min_games: Minimum number of games a strategy must have played
            limit: Maximum number of strategies to return
            
        Returns:
            List of strategy statistics dictionaries
        """
        # Filter strategies with enough games
        filtered_stats = {
            strategy_id: stats 
            for strategy_id, stats in self.strategy_stats.items()
            if stats["games_played"] >= min_games
        }
        
        # Sort by win rate
        sorted_stats = sorted(
            filtered_stats.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )
        
        # Convert to list of dictionaries with strategy_id included
        result = []
        for strategy_id, stats in sorted_stats[:limit]:
            stats_copy = stats.copy()
            stats_copy["strategy_id"] = strategy_id
            result.append(stats_copy)
            
        return result
    
    def generate_win_rate_chart(self, min_games=3):
        """
        Generate a bar chart showing win rates for top strategies.
        
        Args:
            min_games: Minimum number of games a strategy must have played
        """
        top_strategies = self.get_top_strategies(min_games=min_games)
        
        if not top_strategies:
            print("No strategies with enough games played to generate chart.")
            return
            
        # Extract data for plotting
        strategy_ids = [s["strategy_id"] for s in top_strategies]
        win_rates = [s["win_rate"] for s in top_strategies]
        games_played = [s["games_played"] for s in top_strategies]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategy_ids, win_rates)
        
        # Add data labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f'{games_played[i]}g',
                ha='center', va='bottom',
                fontsize=8
            )
        
        # Add labels and title
        plt.xlabel('Strategy ID')
        plt.ylabel('Win Rate')
        plt.title('Strategy Win Rates')
        plt.ylim(0, 1.1)  # Set y-axis from 0 to 1.1 to leave room for labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(self.visualization_dir, 'win_rates.png')
        plt.savefig(chart_path)
        plt.close()
        print(f"Win rate chart saved to {chart_path}")
    
    def generate_evolution_chart(self, top_n=5):
        """
        Generate a line chart showing how win rates evolve over generations.
        
        Args:
            top_n: Number of top strategies to highlight
        """
        # Get all battles sorted by timestamp
        if not self.metrics:
            print("No battle metrics available to generate evolution chart.")
            return
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.metrics)
        
        # Check if we have timestamp information
        if 'timestamp' not in df.columns:
            print("No timestamp information available to generate evolution chart.")
            return
            
        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create a rolling window of strategy performance
        strategy_performance = {}
        
        # Process each battle in chronological order
        for _, battle in df.iterrows():
            # Skip battles without strategy IDs
            if 'team1_strategy_id' not in battle or 'team2_strategy_id' not in battle:
                continue
                
            # Update team1's rolling performance
            team1_id = battle['team1_strategy_id']
            if team1_id not in strategy_performance:
                strategy_performance[team1_id] = {
                    'timestamps': [],
                    'win_rates': [],
                    'games_played': 0,
                    'wins': 0
                }
                
            perf1 = strategy_performance[team1_id]
            perf1['games_played'] += 1
            if battle.get('winner') == battle.get('team1_name'):
                perf1['wins'] += 1
                
            perf1['timestamps'].append(battle['timestamp'])
            perf1['win_rates'].append(perf1['wins'] / perf1['games_played'])
            
            # Update team2's rolling performance
            team2_id = battle['team2_strategy_id']
            if team2_id not in strategy_performance:
                strategy_performance[team2_id] = {
                    'timestamps': [],
                    'win_rates': [],
                    'games_played': 0,
                    'wins': 0
                }
                
            perf2 = strategy_performance[team2_id]
            perf2['games_played'] += 1
            if battle.get('winner') == battle.get('team2_name'):
                perf2['wins'] += 1
                
            perf2['timestamps'].append(battle['timestamp'])
            perf2['win_rates'].append(perf2['wins'] / perf2['games_played'])
        
        # Find top strategies by final win rate
        top_strategies = sorted(
            strategy_performance.items(),
            key=lambda x: x[1]['win_rates'][-1] if x[1]['win_rates'] else 0,
            reverse=True
        )[:top_n]
        
        # Create evolution chart
        plt.figure(figsize=(12, 7))
        
        # Plot each top strategy
        for strategy_id, data in top_strategies:
            if data['win_rates']:
                plt.plot(
                    data['timestamps'], 
                    data['win_rates'], 
                    label=f"{strategy_id} ({data['games_played']} games)"
                )
        
        # Add labels and legend
        plt.xlabel('Time')
        plt.ylabel('Win Rate')
        plt.title('Strategy Win Rate Evolution')
        plt.ylim(0, 1.1)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Save the chart
        chart_path = os.path.join(self.visualization_dir, 'win_rate_evolution.png')
        plt.savefig(chart_path)
        plt.close()
        print(f"Win rate evolution chart saved to {chart_path}")
    
    def generate_strategy_comparison(self):
        """Generate a comparison of different strategy types."""
        # First, identify strategy types from IDs
        strategy_types = {}
        
        for strategy_id in self.strategy_stats:
            # Extract strategy type from id (e.g., "aggressive_abc123" -> "aggressive")
            if "_" in strategy_id:
                strategy_type = strategy_id.split('_')[0]
            else:
                strategy_type = "unknown"
                
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = []
                
            strategy_types[strategy_type].append(strategy_id)
        
        # Calculate average metrics for each strategy type
        type_metrics = {}
        for strategy_type, strategy_ids in strategy_types.items():
            type_metrics[strategy_type] = {
                "count": len(strategy_ids),
                "total_games": 0,
                "avg_win_rate": 0,
                "best_win_rate": 0,
                "best_strategy": None
            }
            
            if not strategy_ids:
                continue
                
            # Calculate total games and win rates
            games_sum = 0
            win_rate_sum = 0
            best_win_rate = 0
            best_strategy = None
            
            for strategy_id in strategy_ids:
                if strategy_id in self.strategy_stats:
                    stats = self.strategy_stats[strategy_id]
                    games_played = stats["games_played"]
                    win_rate = stats["win_rate"]
                    
                    games_sum += games_played
                    win_rate_sum += win_rate * games_played  # weight by games played
                    
                    if win_rate > best_win_rate and games_played >= 3:
                        best_win_rate = win_rate
                        best_strategy = strategy_id
            
            # Calculate averages
            if games_sum > 0:
                type_metrics[strategy_type]["total_games"] = games_sum
                type_metrics[strategy_type]["avg_win_rate"] = win_rate_sum / games_sum
                type_metrics[strategy_type]["best_win_rate"] = best_win_rate
                type_metrics[strategy_type]["best_strategy"] = best_strategy
        
        # Generate comparison chart
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        types = []
        avg_win_rates = []
        best_win_rates = []
        counts = []
        
        for strategy_type, metrics in type_metrics.items():
            if metrics["total_games"] > 0:
                types.append(strategy_type)
                avg_win_rates.append(metrics["avg_win_rate"])
                best_win_rates.append(metrics["best_win_rate"])
                counts.append(metrics["count"])
        
        # Plot grouped bars
        if not types:
            print("No strategy types with games played.")
            return
            
        x = np.arange(len(types))
        width = 0.35
        
        bar1 = plt.bar(x - width/2, avg_win_rates, width, label='Avg Win Rate')
        bar2 = plt.bar(x + width/2, best_win_rates, width, label='Best Win Rate')
        
        # Add labels and title
        plt.xlabel('Strategy Type')
        plt.ylabel('Win Rate')
        plt.title('Strategy Type Comparison')
        plt.xticks(x, types)
        plt.legend()
        
        # Add count annotations
        for i, count in enumerate(counts):
            plt.annotate(f"{count} strats",
                        xy=(x[i], 0.05),
                        ha='center',
                        va='bottom',
                        fontsize=8)
        
        plt.ylim(0, 1.1)
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(self.visualization_dir, 'strategy_comparison.png')
        plt.savefig(chart_path)
        plt.close()
        print(f"Strategy comparison chart saved to {chart_path}")
    
    def generate_reports(self):
        """Generate all available reports and visualizations."""
        self.generate_win_rate_chart()
        self.generate_evolution_chart()
        self.generate_strategy_comparison()
        
        # Print summary report
        print("\n=== Battle Analysis Summary ===")
        print(f"Total battles analyzed: {len(self.metrics)}")
        print(f"Unique strategies: {len(self.strategy_stats)}")
        
        top_strategies = self.get_top_strategies(min_games=3)
        if top_strategies:
            print("\nTop performing strategies:")
            for i, strategy in enumerate(top_strategies[:5]):
                print(f"{i+1}. {strategy['strategy_id']}: " +
                      f"Win Rate {strategy['win_rate']:.2f} " +
                      f"({strategy['wins']}/{strategy['games_played']} games)")
        else:
            print("\nNo strategies with enough games played for ranking.")


if __name__ == "__main__":
    # Example usage
    analyzer = BattleAnalyzer()
    
    # Example adding a battle result
    example_result = {
        "winner": "Team Alpha",
        "team1_name": "Team Alpha",
        "team2_name": "Team Beta",
        "team1_health": 8500,
        "team2_health": 3200,
        "game_duration": 1500,
        "team1_file": "aggressive_abc123.py",
        "team2_file": "defensive_def456.py",
        "message": None,
        "timestamp": "2025-03-18T15:45:12"
    }
    
    analyzer.add_battle_result(example_result)
    
    # Generate reports
    analyzer.generate_reports()