#!/usr/bin/env python3
"""
Tower Defense Reinforcement Learning System - Analysis Tools
---------------------------------------------------------
Provides tools for analyzing training results, visualizing strategy performance,
and generating insights about strategy effectiveness.

Created: 2025-03-20 13:59:39
Author: Mridul-Sharma17
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_data/analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnalysisTools")

class StrategyAnalyzer:
    """
    Analyzes strategy performance and training results to generate insights.
    """
    
    def __init__(self, data_dir: str = "training_data"):
        """
        Initialize the strategy analyzer.
        
        Args:
            data_dir: Directory containing training data
        """
        self.data_dir = data_dir
        self.checkpoint_dir = os.path.join(data_dir, "checkpoints")
        self.results_dir = os.path.join(data_dir, "results")
        self.best_strategies_dir = os.path.join(data_dir, "best_strategies")
        
        # Create output directory for analysis
        self.analysis_dir = os.path.join(data_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Data storage
        self.checkpoints = []
        self.latest_checkpoint = None
        self.all_battle_results = []
        self.all_strategies = {}
        self.metrics_history = {}
    
    def load_data(self) -> bool:
        """
        Load training data from checkpoint files.
        
        Returns:
            True if data was loaded successfully, False otherwise
        """
        # Find all checkpoint files
        checkpoint_files = []
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith("checkpoint_") and filename.endswith(".json"):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    checkpoint_files.append((filepath, os.path.getmtime(filepath)))
        except Exception as e:
            logger.error(f"Error loading checkpoint files: {e}")
            return False
        
        if not checkpoint_files:
            logger.warning("No checkpoint files found")
            return False
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        self.checkpoints = [path for path, _ in checkpoint_files]
        
        # Load the latest checkpoint
        try:
            with open(self.checkpoints[0], 'r') as f:
                self.latest_checkpoint = json.load(f)
                
            # Extract data from checkpoint
            self.all_battle_results = self.latest_checkpoint.get("battle_history", [])
            self.all_strategies = self.latest_checkpoint.get("strategies", {})
            self.metrics_history = self.latest_checkpoint.get("metrics_history", {})
            
            logger.info(f"Loaded data from checkpoint: {self.checkpoints[0]}")
            logger.info(f"Found {len(self.all_battle_results)} battle results and {len(self.all_strategies)} strategies")
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing checkpoint data: {e}")
            return False
    
    def analyze_strategy_evolution(self) -> pd.DataFrame:
        """
        Analyze how strategies have evolved through generations.
        
        Returns:
            DataFrame with strategy evolution data
        """
        evolution_data = []
        
        # Load all checkpoints and extract data by generation
        for checkpoint_path in self.checkpoints:
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                    
                generation = checkpoint.get("generation", 0)
                strategies = checkpoint.get("strategies", {})
                
                # Calculate metrics for this generation
                template_counts = defaultdict(int)
                lane_counts = defaultdict(int)
                avg_win_rate = 0
                highest_win_rate = 0
                strategies_with_games = 0
                
                for strategy in strategies.values():
                    template = strategy.get("template_type", "unknown")
                    lane = strategy.get("lane_preference", "unknown")
                    metrics = strategy.get("metrics", {})
                    games_played = metrics.get("games_played", 0)
                    
                    template_counts[template] += 1
                    lane_counts[lane] += 1
                    
                    if games_played > 0:
                        win_rate = metrics.get("win_rate", 0)
                        avg_win_rate += win_rate
                        highest_win_rate = max(highest_win_rate, win_rate)
                        strategies_with_games += 1
                
                if strategies_with_games > 0:
                    avg_win_rate /= strategies_with_games
                
                # Find most popular template and lane
                most_popular_template = max(template_counts.items(), key=lambda x: x[1])[0] if template_counts else "unknown"
                most_popular_lane = max(lane_counts.items(), key=lambda x: x[1])[0] if lane_counts else "unknown"
                
                # Add generation data
                evolution_data.append({
                    "Generation": generation,
                    "Strategy Count": len(strategies),
                    "Avg Win Rate": avg_win_rate,
                    "Highest Win Rate": highest_win_rate,
                    "Most Popular Template": most_popular_template,
                    "Most Popular Lane": most_popular_lane,
                    "Diversity": checkpoint.get("metrics_history", {}).get("diversity", [])[-1] if checkpoint.get("metrics_history", {}).get("diversity") else 0
                })
                
            except Exception as e:
                logger.warning(f"Error processing checkpoint {checkpoint_path}: {e}")
        
        # Convert to DataFrame and sort by generation
        df = pd.DataFrame(evolution_data)
        if not df.empty:
            df = df.sort_values("Generation")
        
        return df
    
    def analyze_template_performance(self) -> pd.DataFrame:
        """
        Analyze performance of different strategy templates.
        
        Returns:
            DataFrame with template performance data
        """
        if not self.all_strategies:
            logger.warning("No strategy data available")
            return pd.DataFrame()
        
        template_data = defaultdict(lambda: {
            "count": 0,
            "win_sum": 0,
            "loss_sum": 0,
            "games_sum": 0,
            "win_rate_sum": 0,
            "health_sum": 0
        })
        
        # Aggregate data by template
        for strategy in self.all_strategies.values():
            template = strategy.get("template_type", "unknown")
            metrics = strategy.get("metrics", {})
            games_played = metrics.get("games_played", 0)
            
            template_data[template]["count"] += 1
            
            if games_played > 0:
                wins = metrics.get("wins", 0)
                losses = metrics.get("losses", 0)
                win_rate = metrics.get("win_rate", 0)
                health = metrics.get("avg_tower_health", 0)
                
                template_data[template]["win_sum"] += wins
                template_data[template]["loss_sum"] += losses
                template_data[template]["games_sum"] += games_played
                template_data[template]["win_rate_sum"] += win_rate
                template_data[template]["health_sum"] += health
        
        # Calculate averages and create dataframe
        performance_data = []
        
        for template, data in template_data.items():
            count = data["count"]
            games_sum = data["games_sum"]
            
            if count > 0 and games_sum > 0:
                avg_games = games_sum / count
                avg_win_rate = data["win_rate_sum"] / count
                avg_health = data["health_sum"] / count if count > 0 else 0
                
                performance_data.append({
                    "Template": template,
                    "Strategy Count": count,
                    "Total Games": games_sum,
                    "Total Wins": data["win_sum"],
                    "Total Losses": data["loss_sum"],
                    "Avg Games Per Strategy": avg_games,
                    "Avg Win Rate": avg_win_rate,
                    "Avg Tower Health": avg_health
                })
        
        # Convert to DataFrame and sort by win rate
        df = pd.DataFrame(performance_data)
        if not df.empty:
            df = df.sort_values("Avg Win Rate", ascending=False)
        
        return df
    
    def analyze_parameter_impact(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze the impact of different parameters on strategy performance.
        
        Returns:
            Dictionary of DataFrames with parameter impact analysis
        """
        if not self.all_strategies:
            logger.warning("No strategy data available")
            return {}
        
        # Parameters to analyze
        parameter_analyses = {}
        
        # Filter strategies with enough games
        valid_strategies = []
        for strategy_id, strategy in self.all_strategies.items():
            metrics = strategy.get("metrics", {})
            games_played = metrics.get("games_played", 0)
            
            if games_played >= 5:  # Require at least 5 games for meaningful analysis
                valid_strategies.append((strategy_id, strategy))
        
        if not valid_strategies:
            logger.warning("No strategies with enough games for parameter analysis")
            return {}
        
        # Analyze categorical parameters
        categorical_params = [
            ("template_type", "Template Type"), 
            ("lane_preference", "Lane Preference")
        ]
        
        for param_key, param_name in categorical_params:
            analysis = self._analyze_categorical_param(valid_strategies, param_key, param_name)
            if not analysis.empty:
                parameter_analyses[param_key] = analysis
        
        # Analyze numeric parameters
        numeric_params = [
            ("defensive_trigger", "Defensive Trigger"),
            ("elixir_thresholds.min_deploy", "Min Deploy Elixir"),
            ("elixir_thresholds.save_threshold", "Save Threshold"),
            ("counter_weights.air_vs_ground", "Air vs Ground Weight"),
            ("phase_parameters.early_game_aggression", "Early Game Aggression"),
            ("phase_parameters.late_game_aggression", "Late Game Aggression")
        ]
        
        for param_key, param_name in numeric_params:
            analysis = self._analyze_numeric_param(valid_strategies, param_key, param_name)
            if not analysis.empty:
                parameter_analyses[param_key] = analysis
        
        return parameter_analyses
    
    def _analyze_categorical_param(self, 
                                  strategies: List[Tuple[str, Dict[str, Any]]], 
                                  param_key: str, 
                                  param_name: str) -> pd.DataFrame:
        """Analyze impact of a categorical parameter."""
        param_data = defaultdict(lambda: {
            "count": 0,
            "win_rate_sum": 0,
            "health_sum": 0,
            "games_sum": 0
        })
        
        for _, strategy in strategies:
            value = strategy.get(param_key, "unknown")
            metrics = strategy.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            health = metrics.get("avg_tower_health", 0)
            games = metrics.get("games_played", 0)
            
            param_data[value]["count"] += 1
            param_data[value]["win_rate_sum"] += win_rate
            param_data[value]["health_sum"] += health
            param_data[value]["games_sum"] += games
        
        analysis_data = []
        
        for value, data in param_data.items():
            count = data["count"]
            if count > 0:
                avg_win_rate = data["win_rate_sum"] / count
                avg_health = data["health_sum"] / count
                avg_games = data["games_sum"] / count
                
                analysis_data.append({
                    param_name: value,
                    "Strategy Count": count,
                    "Avg Win Rate": avg_win_rate,
                    "Avg Tower Health": avg_health,
                    "Avg Games": avg_games
                })
        
        df = pd.DataFrame(analysis_data)
        if not df.empty:
            df = df.sort_values("Avg Win Rate", ascending=False)
        
        return df
    
    def _analyze_numeric_param(self, 
                              strategies: List[Tuple[str, Dict[str, Any]]], 
                              param_key: str, 
                              param_name: str) -> pd.DataFrame:
        """Analyze impact of a numeric parameter."""
        # Extract nested parameters if needed
        if "." in param_key:
            parts = param_key.split(".")
            get_value = lambda s: self._get_nested_value(s, parts)
        else:
            get_value = lambda s: s.get(param_key)
        
        # Collect parameter values and performance metrics
        param_values = []
        
        for _, strategy in strategies:
            value = get_value(strategy)
            if value is not None:
                metrics = strategy.get("metrics", {})
                win_rate = metrics.get("win_rate", 0)
                health = metrics.get("avg_tower_health", 0)
                games = metrics.get("games_played", 0)
                
                param_values.append({
                    param_name: value,
                    "Win Rate": win_rate,
                    "Tower Health": health,
                    "Games Played": games
                })
        
        if not param_values:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(param_values)
        
        # Create 5 equal-width bins for the parameter
        df["Bin"] = pd.cut(df[param_name], bins=5)
        
        # Aggregate by bin
        bin_stats = df.groupby("Bin").agg({
            param_name: "mean",
            "Win Rate": "mean",
            "Tower Health": "mean",
            "Games Played": ["count", "sum"]
        })
        
        # Flatten multi-level columns
        bin_stats.columns = [f"{a}_{b}" if b else a for a, b in bin_stats.columns]
        
        # Reset index to make Bin a column
        bin_stats = bin_stats.reset_index()
        
        # Rename columns for clarity
        bin_stats = bin_stats.rename(columns={
            f"{param_name}_mean": param_name,
            "Win Rate_mean": "Avg Win Rate",
            "Tower Health_mean": "Avg Tower Health",
            "Games Played_count": "Strategy Count",
            "Games Played_sum": "Total Games"
        })
        
        return bin_stats
    
    def _get_nested_value(self, strategy: Dict[str, Any], key_parts: List[str]) -> Optional[Any]:
        """Get a value from a nested dictionary structure."""
        current = strategy
        for part in key_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    
    def analyze_battle_results(self) -> Dict[str, Any]:
        """
        Analyze the results of all battles.
        
        Returns:
            Dictionary with battle analysis results
        """
        if not self.all_battle_results:
            logger.warning("No battle results available")
            return {}
        
        # Collect basic stats
        total_battles = len(self.all_battle_results)
        
        # Winner/loser stats
        winner_counts = Counter(battle.get("winner") for battle in self.all_battle_results if "winner" in battle)
        loser_counts = Counter(battle.get("loser") for battle in self.all_battle_results if "loser" in battle)
        
        # Calculate win rates for strategies with at least 5 battles
        strategy_battles = defaultdict(int)
        strategy_wins = defaultdict(int)
        
        for battle in self.all_battle_results:
            winner = battle.get("winner")
            loser = battle.get("loser")
            
            if winner:
                strategy_battles[winner] += 1
                strategy_wins[winner] += 1
            
            if loser:
                strategy_battles[loser] += 1
        
        # Calculate win rates
        win_rates = {}
        for strategy_id, battles in strategy_battles.items():
            if battles >= 5:  # Only calculate for strategies with at least 5 battles
                wins = strategy_wins.get(strategy_id, 0)
                win_rates[strategy_id] = wins / battles
        
        # Health stats
        winner_health = [battle.get("winner_tower_health", 0) for battle in self.all_battle_results]
        loser_health = [battle.get("loser_tower_health", 0) for battle in self.all_battle_results]
        
        # Game duration stats
        durations = [battle.get("game_duration", 0) for battle in self.all_battle_results]
        
        # Matchup data
        matchups = []
        for battle in self.all_battle_results:
            winner = battle.get("winner")
            loser = battle.get("loser")
            
            if winner and loser and winner in self.all_strategies and loser in self.all_strategies:
                winner_template = self.all_strategies[winner].get("template_type", "unknown")
                loser_template = self.all_strategies[loser].get("template_type", "unknown")
                
                matchups.append((winner_template, loser_template))
        
        # Create matchup matrix
        matchup_counts = defaultdict(lambda: defaultdict(int))
        
        for winner_template, loser_template in matchups:
            matchup_counts[winner_template][loser_template] += 1
        
        # Convert to dataframe
        template_types = sorted(set(template for winner_dict in matchup_counts.values() 
                                   for template in winner_dict.keys()) | 
                               set(matchup_counts.keys()))
        
        matchup_matrix = []
        for row_template in template_types:
            row = []
            for col_template in template_types:
                row.append(matchup_counts[row_template][col_template])
            matchup_matrix.append(row)
        
        matchup_df = pd.DataFrame(matchup_matrix, index=template_types, columns=template_types)
        
        # Return all analysis results
        return {
            "total_battles": total_battles,
            "top_winners": winner_counts.most_common(5),
            "most_losses": loser_counts.most_common(5),
            "win_rates": sorted(win_rates.items(), key=lambda x: x[1], reverse=True)[:10],
            "avg_winner_health": np.mean(winner_health) if winner_health else 0,
            "avg_loser_health": np.mean(loser_health) if loser_health else 0,
            "avg_game_duration": np.mean(durations) if durations else 0,
            "matchup_matrix": matchup_df
        }
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate analysis reports and visualizations.
        
        Returns:
            Dictionary mapping report types to file paths
        """
        if not self.load_data():
            logger.error("Failed to load data for analysis")
            return {}
        
        report_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Strategy Evolution Report
            evolution_data = self.analyze_strategy_evolution()
            if not evolution_data.empty:
                # Save to CSV
                evolution_csv = os.path.join(self.analysis_dir, f"evolution_report_{timestamp}.csv")
                evolution_data.to_csv(evolution_csv, index=False)
                report_files["evolution_csv"] = evolution_csv
                
                # Create visualizations
                evolution_plot = os.path.join(self.analysis_dir, f"evolution_plot_{timestamp}.png")
                self._plot_evolution(evolution_data, evolution_plot)
                report_files["evolution_plot"] = evolution_plot
            
            # 2. Template Performance Report
            template_data = self.analyze_template_performance()
            if not template_data.empty:
                # Save to CSV
                template_csv = os.path.join(self.analysis_dir, f"template_performance_{timestamp}.csv")
                template_data.to_csv(template_csv, index=False)
                report_files["template_csv"] = template_csv
                
                # Create visualizations
                template_plot = os.path.join(self.analysis_dir, f"template_performance_{timestamp}.png")
                self._plot_template_performance(template_data, template_plot)
                report_files["template_plot"] = template_plot
            
            # 3. Parameter Impact Analysis
            param_analyses = self.analyze_parameter_impact()
            if param_analyses:
                # Create parameter impact visualizations
                param_plot = os.path.join(self.analysis_dir, f"parameter_impact_{timestamp}.png")
                self._plot_parameter_impact(param_analyses, param_plot)
                report_files["parameter_plot"] = param_plot
                
                # Save individual parameter analyses to CSV
                for param_key, df in param_analyses.items():
                    param_file = os.path.join(self.analysis_dir, f"param_{param_key}_{timestamp}.csv")
                    df.to_csv(param_file, index=False)
                    report_files[f"param_{param_key}_csv"] = param_file
            
            # 4. Battle Results Analysis
            battle_analysis = self.analyze_battle_results()
            if battle_analysis:
                # Create matchup heatmap
                matchup_plot = os.path.join(self.analysis_dir, f"matchup_heatmap_{timestamp}.png")
                self._plot_matchup_heatmap(battle_analysis.get("matchup_matrix"), matchup_plot)
                report_files["matchup_plot"] = matchup_plot
                
                # Create battle summary
                battle_summary = os.path.join(self.analysis_dir, f"battle_summary_{timestamp}.txt")
                self._create_battle_summary(battle_analysis, battle_summary)
                report_files["battle_summary"] = battle_summary
            
            # 5. Generate comprehensive report
            comprehensive_report = os.path.join(self.analysis_dir, f"comprehensive_report_{timestamp}.html")
            self._create_comprehensive_report(
                evolution_data, 
                template_data, 
                param_analyses,
                battle_analysis, 
                comprehensive_report
            )
            report_files["comprehensive_report"] = comprehensive_report
            
            logger.info(f"Generated {len(report_files)} analysis reports in {self.analysis_dir}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}", exc_info=True)
        
        return report_files
    
    def _plot_evolution(self, data: pd.DataFrame, output_path: str):
        """Create a visualization of strategy evolution over generations."""
        plt.figure(figsize=(12, 8))
        
        # Create a multiple subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Win Rate Evolution
        axes[0, 0].plot(data["Generation"], data["Avg Win Rate"], label="Average Win Rate", marker='o')
        axes[0, 0].plot(data["Generation"], data["Highest Win Rate"], label="Highest Win Rate", marker='x')
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Win Rate")
        axes[0, 0].set_title("Win Rate Evolution")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Population Diversity
        axes[0, 1].plot(data["Generation"], data["Diversity"], color='green', marker='o')
        axes[0, 1].set_xlabel("Generation")
        axes[0, 1].set_ylabel("Diversity")
        axes[0, 1].set_title("Population Diversity")
        axes[0, 1].grid(True)
        
        # Most Popular Template by Generation
        if len(data) > 1:  # Only if we have more than one generation
            template_pivot = pd.pivot_table(
                data,
                index="Generation",
                columns="Most Popular Template",
                aggfunc=lambda x: 1,
                fill_value=0
            )
            
            # Plotting a stacked area plot would be ideal, but it's complex with this data
            # Instead, we'll create a scatter plot with custom markers for each template
            unique_templates = data["Most Popular Template"].unique()
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            
            for i, template in enumerate(unique_templates):
                template_gens = data[data["Most Popular Template"] == template]["Generation"]
                if len(template_gens) > 0:
                    marker = markers[i % len(markers)]
                    axes[1, 0].scatter(template_gens, [i] * len(template_gens), 
                                    marker=marker, s=100, label=template)
            
            axes[1, 0].set_yticks(range(len(unique_templates)))
            axes[1, 0].set_yticklabels(unique_templates)
            axes[1, 0].set_xlabel("Generation")
            axes[1, 0].set_title("Dominant Template Type by Generation")
            axes[1, 0].legend(loc='upper right')
            axes[1, 0].grid(True)
        
        # Most Popular Lane by Generation
        if len(data) > 1:  # Only if we have more than one generation
            lane_counts = defaultdict(list)
            
            for _, row in data.iterrows():
                lane_counts[row["Most Popular Lane"]].append(row["Generation"])
            
            for lane, generations in lane_counts.items():
                axes[1, 1].plot(generations, [1] * len(generations), 'o', label=lane, markersize=10)
            
            axes[1, 1].set_yticks([])
            axes[1, 1].set_xlabel("Generation")
            axes[1, 1].set_title("Dominant Lane Preference by Generation")
            if lane_counts:
                axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_template_performance(self, data: pd.DataFrame, output_path: str):
        """Create a visualization of template performance."""
        plt.figure(figsize=(12, 10))
        
        # Create bar charts
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Win Rate by Template
        win_bars = axes[0].bar(data["Template"], data["Avg Win Rate"], color='skyblue')
        axes[0].set_xlabel("Template Type")
        axes[0].set_ylabel("Average Win Rate")
        axes[0].set_title("Average Win Rate by Template Type")
        axes[0].set_ylim(0, max(data["Avg Win Rate"]) * 1.2)
        axes[0].grid(axis='y')
        
        # Add values above bars
        for bar in win_bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Remaining Tower Health by Template
        health_bars = axes[1].bar(data["Template"], data["Avg Tower Health"], color='lightgreen')
        axes[1].set_xlabel("Template Type")
        axes[1].set_ylabel("Average Tower Health")
        axes[1].set_title("Average Remaining Tower Health by Template Type")
        axes[1].grid(axis='y')
        
        # Add values above bars
        for bar in health_bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 50,
                      f'{height:.0f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_parameter_impact(self, param_analyses: Dict[str, pd.DataFrame], output_path: str):
        """Create visualizations of parameter impact on performance."""
        # Determine number of parameters to plot
        num_params = len(param_analyses)
        
        if num_params == 0:
            logger.warning("No parameter data to plot")
            return
        
        # Determine grid size: Try to make it as square as possible
        grid_size = int(np.ceil(np.sqrt(num_params)))
        rows = grid_size
        cols = grid_size if grid_size * (grid_size - 1) < num_params else grid_size - 1
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = axes.flatten() if num_params > 1 else [axes]
        
        # Plot each parameter's impact
        for i, (param_key, df) in enumerate(param_analyses.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                if param_key in ["template_type", "lane_preference"]:
                    # Categorical parameter
                    param_name = "Template Type" if param_key == "template_type" else "Lane Preference"
                    bars = ax.bar(df[param_name], df["Avg Win Rate"], color='skyblue')
                    ax.set_xlabel(param_name)
                    ax.set_ylabel("Avg Win Rate")
                    ax.set_title(f"{param_name} Impact on Win Rate")
                    
                    # Add values above bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom', rotation=0)
                    
                    # Set strategy count as text below x-axis
                    for i, v in enumerate(df["Strategy Count"]):
                        ax.text(i, -0.05, f'n={v}', ha='center', va='top', transform=ax.get_xaxis_transform())
                else:
                    # Numeric parameter
                    # Extract param name from key
                    if "." in param_key:
                        param_name = param_key.split(".")[-1].replace("_", " ").title()
                    else:
                        param_name = param_key.replace("_", " ").title()
                    
                    # Plot line chart for numeric parameters
                    line = ax.plot(df[param_name], df["Avg Win Rate"], marker='o', linestyle='-', color='blue')
                    
                    # Add scatter points with size based on strategy count
                    sizes = df["Strategy Count"] * 20  # Scale for visibility
                    scatter = ax.scatter(df[param_name], df["Avg Win Rate"], s=sizes, alpha=0.6)
                    
                    ax.set_xlabel(param_name)
                    ax.set_ylabel("Avg Win Rate")
                    ax.set_title(f"Impact of {param_name} on Win Rate")
                    ax.grid(True)
                    
                    # Add annotations for important points
                    max_idx = df["Avg Win Rate"].idxmax()
                    if not pd.isna(max_idx):
                        max_x = df.loc[max_idx, param_name]
                        max_y = df.loc[max_idx, "Avg Win Rate"]
                        ax.annotate(f'Best: {max_y:.3f}',
                                  xy=(max_x, max_y),
                                  xytext=(5, 10),
                                  textcoords='offset points',
                                  arrowprops=dict(arrowstyle='->'))
            except Exception as e:
                logger.warning(f"Error plotting parameter {param_key}: {e}")
                ax.text(0.5, 0.5, f"Error plotting {param_key}",
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide unused subplots
        for i in range(num_params, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_matchup_heatmap(self, matchup_df: pd.DataFrame, output_path: str):
        """Create a heatmap visualization of template matchups."""
        if matchup_df is None or matchup_df.empty:
            logger.warning("No matchup data to plot")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create matchup heatmap
        ax = sns.heatmap(matchup_df, annot=True, cmap="YlGnBu", fmt="d",
                       cbar_kws={'label': 'Number of Wins'})
        
        # Set labels and title
        plt.xlabel("Losing Template")
        plt.ylabel("Winning Template")
        plt.title("Strategy Template Matchup Heatmap")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _create_battle_summary(self, battle_analysis: Dict[str, Any], output_path: str):
        """Create a text summary of battle results."""
        if not battle_analysis:
            logger.warning("No battle data for summary")
            return
        
        try:
            with open(output_path, 'w') as f:
                f.write("=== Tower Defense Training Battle Summary ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Overall statistics
                f.write("== Overall Statistics ==\n")
                f.write(f"Total Battles: {battle_analysis.get('total_battles', 0)}\n")
                f.write(f"Average Game Duration: {battle_analysis.get('avg_game_duration', 0):.1f} frames\n")
                f.write(f"Average Winner Tower Health: {battle_analysis.get('avg_winner_health', 0):.1f}\n")
                f.write(f"Average Loser Tower Health: {battle_analysis.get('avg_loser_health', 0):.1f}\n\n")
                
                # Top winning strategies
                f.write("== Top Winning Strategies ==\n")
                for i, (strategy_id, wins) in enumerate(battle_analysis.get('top_winners', []), 1):
                    template = "unknown"
                    if strategy_id in self.all_strategies:
                        template = self.all_strategies[strategy_id].get("template_type", "unknown")
                    f.write(f"{i}. Strategy {strategy_id}: {wins} wins - Template: {template}\n")
                f.write("\n")
                
                # Most defeated strategies
                f.write("== Most Defeated Strategies ==\n")
                for i, (strategy_id, losses) in enumerate(battle_analysis.get('most_losses', []), 1):
                    template = "unknown"
                    if strategy_id in self.all_strategies:
                        template = self.all_strategies[strategy_id].get("template_type", "unknown")
                    f.write(f"{i}. Strategy {strategy_id}: {losses} losses - Template: {template}\n")
                f.write("\n")
                
                # Best win rates
                f.write("== Best Win Rates (min. 5 battles) ==\n")
                for i, (strategy_id, win_rate) in enumerate(battle_analysis.get('win_rates', []), 1):
                    template = "unknown"
                    if strategy_id in self.all_strategies:
                        template = self.all_strategies[strategy_id].get("template_type", "unknown")
                        games = self.all_strategies[strategy_id].get("metrics", {}).get("games_played", 0)
                    else:
                        games = 0
                    f.write(f"{i}. Strategy {strategy_id}: {win_rate:.3f} win rate ({games} games) - Template: {template}\n")
                f.write("\n")
                
                # Template matchups summary
                f.write("== Template Matchup Summary ==\n")
                matchup_df = battle_analysis.get("matchup_matrix")
                if matchup_df is not None and not matchup_df.empty:
                    f.write("(Rows are winning templates, columns are losing templates)\n\n")
                    f.write(matchup_df.to_string())
                else:
                    f.write("No matchup data available\n")
                
                f.write("\n\n== End of Battle Summary ==\n")
            
            logger.info(f"Created battle summary at {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating battle summary: {e}")
    
    def _create_comprehensive_report(self,
                                   evolution_data: pd.DataFrame,
                                   template_data: pd.DataFrame,
                                   param_analyses: Dict[str, pd.DataFrame],
                                   battle_analysis: Dict[str, Any],
                                   output_path: str):
        """Create a comprehensive HTML report with all analysis results."""
        try:
            with open(output_path, 'w') as f:
                # HTML header
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tower Defense Training Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #0066cc;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .summary-box {
            background-color: #f0f0f0;
            border-left: 4px solid #0066cc;
            padding: 10px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Tower Defense Training Analysis Report</h1>
    <p>Generated: """)
                
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
                
                # Summary section
                f.write('<div class="section">\n')
                f.write('<h2>Executive Summary</h2>\n')
                f.write('<div class="summary-box">\n')
                
                if not evolution_data.empty:
                    last_gen = evolution_data.iloc[-1]
                    total_gens = int(last_gen["Generation"]) + 1
                    best_win_rate = last_gen["Highest Win Rate"]
                    f.write(f'<p>Training completed <strong>{total_gens}</strong> generations.</p>\n')
                    f.write(f'<p>Best strategy achieved a win rate of <strong>{best_win_rate:.3f}</strong>.</p>\n')
                
                if battle_analysis:
                    f.write(f'<p>Total battles: <strong>{battle_analysis.get("total_battles", 0)}</strong></p>\n')
                    f.write(f'<p>Average game duration: <strong>{battle_analysis.get("avg_game_duration", 0):.1f}</strong> frames</p>\n')
                
                if template_data is not None and not template_data.empty:
                    best_template = template_data.iloc[0]["Template"]
                    best_template_win_rate = template_data.iloc[0]["Avg Win Rate"]
                    f.write(f'<p>Best performing template: <strong>{best_template}</strong> with {best_template_win_rate:.3f} win rate</p>\n')
                
                f.write('</div>\n')
                f.write('</div>\n')
                
                # Strategy Evolution Section
                f.write('<div class="section">\n')
                f.write('<h2>Strategy Evolution</h2>\n')
                
                if not evolution_data.empty:
                    f.write('<h3>Evolution Metrics</h3>\n')
                    f.write('<table>\n')
                    f.write('  <tr>\n')
                    for col in evolution_data.columns:
                        f.write(f'    <th>{col}</th>\n')
                    f.write('  </tr>\n')
                    
                    for _, row in evolution_data.iterrows():
                        f.write('  <tr>\n')
                        for col in evolution_data.columns:
                            value = row[col]
                            if isinstance(value, float):
                                f.write(f'    <td>{value:.4f}</td>\n')
                            else:
                                f.write(f'    <td>{value}</td>\n')
                        f.write('  </tr>\n')
                    
                    f.write('</table>\n')
                else:
                    f.write('<p>No evolution data available.</p>\n')
                
                f.write('</div>\n')
                
                # Template Performance Section
                f.write('<div class="section">\n')
                f.write('<h2>Template Performance</h2>\n')
                
                if not template_data.empty:
                    f.write('<h3>Template Metrics</h3>\n')
                    f.write('<table>\n')
                    f.write('  <tr>\n')
                    for col in template_data.columns:
                        f.write(f'    <th>{col}</th>\n')
                    f.write('  </tr>\n')
                    
                    for _, row in template_data.iterrows():
                        f.write('  <tr>\n')
                        for col in template_data.columns:
                            value = row[col]
                            if isinstance(value, float):
                                f.write(f'    <td>{value:.4f}</td>\n')
                            else:
                                f.write(f'    <td>{value}</td>\n')
                        f.write('  </tr>\n')
                    
                    f.write('</table>\n')
                else:
                    f.write('<p>No template performance data available.</p>\n')
                
                f.write('</div>\n')
                
                # Parameter Impact Section
                f.write('<div class="section">\n')
                f.write('<h2>Parameter Impact Analysis</h2>\n')
                
                if param_analyses:
                    for param_key, df in param_analyses.items():
                        # Format parameter name for display
                        if "." in param_key:
                            parts = param_key.split(".")
                            param_display = parts[-1].replace("_", " ").title()
                            param_category = parts[0].replace("_", " ").title()
                            param_title = f"{param_category} - {param_display}"
                        else:
                            param_title = param_key.replace("_", " ").title()
                        
                        f.write(f'<h3>{param_title}</h3>\n')
                        f.write('<table>\n')
                        f.write('  <tr>\n')
                        for col in df.columns:
                            f.write(f'    <th>{col}</th>\n')
                        f.write('  </tr>\n')
                        
                        for _, row in df.iterrows():
                            f.write('  <tr>\n')
                            for col in df.columns:
                                value = row[col]
                                if isinstance(value, float):
                                    f.write(f'    <td>{value:.4f}</td>\n')
                                else:
                                    f.write(f'    <td>{value}</td>\n')
                            f.write('  </tr>\n')
                        
                        f.write('</table>\n')
                else:
                    f.write('<p>No parameter analysis data available.</p>\n')
                
                f.write('</div>\n')
                
                # Battle Results Section
                f.write('<div class="section">\n')
                f.write('<h2>Battle Results</h2>\n')
                
                if battle_analysis:
                    # Overall statistics
                    f.write('<h3>Battle Statistics</h3>\n')
                    f.write('<ul>\n')
                    f.write(f'  <li>Total Battles: {battle_analysis.get("total_battles", 0)}</li>\n')
                    f.write(f'  <li>Average Game Duration: {battle_analysis.get("avg_game_duration", 0):.1f} frames</li>\n')
                    f.write(f'  <li>Average Winner Tower Health: {battle_analysis.get("avg_winner_health", 0):.1f}</li>\n')
                    f.write(f'  <li>Average Loser Tower Health: {battle_analysis.get("avg_loser_health", 0):.1f}</li>\n')
                    f.write('</ul>\n')
                    
                    # Top winning strategies
                    f.write('<h3>Top Winning Strategies</h3>\n')
                    f.write('<table>\n')
                    f.write('  <tr><th>Rank</th><th>Strategy ID</th><th>Template</th><th>Wins</th></tr>\n')
                    
                    for i, (strategy_id, wins) in enumerate(battle_analysis.get('top_winners', []), 1):
                        template = "unknown"
                        if strategy_id in self.all_strategies:
                            template = self.all_strategies[strategy_id].get("template_type", "unknown")
                        f.write(f'  <tr><td>{i}</td><td>{strategy_id}</td><td>{template}</td><td>{wins}</td></tr>\n')
                    
                    f.write('</table>\n')
                    
                    # Best win rates
                    f.write('<h3>Best Win Rates (min. 5 battles)</h3>\n')
                    f.write('<table>\n')
                    f.write('  <tr><th>Rank</th><th>Strategy ID</th><th>Template</th><th>Win Rate</th><th>Games</th></tr>\n')
                    
                    for i, (strategy_id, win_rate) in enumerate(battle_analysis.get('win_rates', []), 1):
                        template = "unknown"
                        if strategy_id in self.all_strategies:
                            template = self.all_strategies[strategy_id].get("template_type", "unknown")
                            games = self.all_strategies[strategy_id].get("metrics", {}).get("games_played", 0)
                        else:
                            games = 0
                        f.write(f'  <tr><td>{i}</td><td>{strategy_id}</td><td>{template}</td><td>{win_rate:.3f}</td><td>{games}</td></tr>\n')
                    
                    f.write('</table>\n')
                    
                    # Template matchups
                    f.write('<h3>Template Matchup Matrix</h3>\n')
                    matchup_df = battle_analysis.get("matchup_matrix")
                    if matchup_df is not None and not matchup_df.empty:
                        f.write('<p>(Rows are winning templates, columns are losing templates)</p>\n')
                        f.write('<table>\n')
                        
                        # Header row with column names
                        f.write('  <tr><th></th>\n')
                        for col in matchup_df.columns:
                            f.write(f'    <th>{col}</th>\n')
                        f.write('  </tr>\n')
                        
                        # Data rows
                        for idx, row in matchup_df.iterrows():
                            f.write(f'  <tr><th>{idx}</th>\n')
                            for col in matchup_df.columns:
                                f.write(f'    <td>{row[col]}</td>\n')
                            f.write('  </tr>\n')
                        
                        f.write('</table>\n')
                    else:
                        f.write('<p>No matchup data available.</p>\n')
                else:
                    f.write('<p>No battle analysis data available.</p>\n')
                
                f.write('</div>\n')
                
                # Footer
                f.write("""
    <div style="margin-top: 50px; text-align: center; font-size: 0.8em; color: #666;">
        <p>Tower Defense Reinforcement Learning System Analysis Report</p>
        <p>Generated by StrategyAnalyzer</p>
    </div>
</body>
</html>
""")
            
            logger.info(f"Created comprehensive HTML report at {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating comprehensive report: {e}")


# Command-line interface for running analysis directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Tower Defense training results")
    parser.add_argument("--data-dir", type=str, default="training_data",
                      help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory for analysis output (defaults to data_dir/analysis)")
    
    args = parser.parse_args()
    
    # Set up analyzer
    analyzer = StrategyAnalyzer(data_dir=args.data_dir)
    
    if args.output_dir:
        analyzer.analysis_dir = args.output_dir
        os.makedirs(analyzer.analysis_dir, exist_ok=True)
    
    # Generate reports
    print(f"Analyzing training data in {args.data_dir}...")
    report_files = analyzer.generate_reports()
    
    if report_files:
        print(f"Generated {len(report_files)} analysis files:")
        for report_type, file_path in report_files.items():
            print(f"  - {report_type}: {file_path}")
    else:
        print("No reports generated. Check if training data exists.")