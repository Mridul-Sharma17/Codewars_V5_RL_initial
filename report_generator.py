"""
Tower Defense RL - Report Generator

This module generates HTML reports and visualizations for strategy evolution.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from config import *

logger = logging.getLogger('report_generator')

class ReportGenerator:
    """Generates HTML reports and visualizations for strategy evolution."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.report_dir = REPORT_DIR
        self.visualization_dir = VISUALIZATION_DIR
        
        # Ensure directories exist
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
    
    def generate_generation_report(
        self, 
        generation: int, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]],
        fitness_scores: Dict[str, float]
    ) -> str:
        """
        Generate a report for a specific generation.
        
        Args:
            generation: Generation number
            strategies: Dictionary of strategy_id -> strategy_data
            battle_metrics: Dictionary of battle_id -> battle_data
            fitness_scores: Dictionary of strategy_id -> fitness_score
            
        Returns:
            Path to the generated report
        """
        # Create directory for this generation
        gen_dir = os.path.join(self.report_dir, f"generation_{generation}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Generate visualizations
        self._generate_win_rate_chart(generation, strategies, fitness_scores)
        self._generate_fitness_components_chart(generation, strategies)
        
        # Generate report data
        report_data = self._prepare_report_data(generation, strategies, battle_metrics, fitness_scores)
        
        # Generate HTML report
        report_path = os.path.join(gen_dir, "report.html")
        self._generate_html_report(report_path, report_data)
        
        logger.info(f"Generated report for generation {generation} at {report_path}")
        return report_path
    
    def generate_final_report(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate a final report summarizing all generations.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_data
            battle_metrics: Dictionary of battle_id -> battle_data
            
        Returns:
            Path to the generated report
        """
        # Generate performance evolution chart
        self._generate_performance_evolution_chart(strategies)
        self._generate_archetype_comparison_chart(strategies)
        
        # Prepare report data
        report_data = self._prepare_final_report_data(strategies, battle_metrics)
        
        # Generate HTML report
        report_path = os.path.join(self.report_dir, "final_report.html")
        self._generate_final_html_report(report_path, report_data)
        
        logger.info(f"Generated final report at {report_path}")
        return report_path
    
    def _prepare_report_data(
        self, 
        generation: int, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]],
        fitness_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Prepare data for the generation report."""
        # Sort strategies by fitness score
        sorted_strategies = sorted(
            [(s_id, s) for s_id, s in strategies.items()],
            key=lambda x: fitness_scores.get(x[0], 0),
            reverse=True
        )
        
        # Extract strategy summary
        strategy_summary = []
        for strategy_id, strategy in sorted_strategies:
            summary = {
                "id": strategy_id,
                "name": strategy.get("name", strategy_id),
                "template_type": strategy.get("template_type", "unknown"),
                "generation": strategy.get("generation", 0),
                "games_played": strategy.get("metrics", {}).get("games_played", 0),
                "wins": strategy.get("metrics", {}).get("wins", 0),
                "losses": strategy.get("metrics", {}).get("losses", 0),
                "win_rate": strategy.get("metrics", {}).get("win_rate", 0),
                "avg_tower_health": strategy.get("metrics", {}).get("avg_tower_health", 0),
                "fitness_score": fitness_scores.get(strategy_id, 0)
            }
            strategy_summary.append(summary)
        
        # Extract battle summary (top 10 most decisive battles)
        battle_summary = []
        for battle_id, battle in battle_metrics.items():
            if battle.get("strategy1_health") is None or battle.get("strategy2_health") is None:
                continue
            
            # Calculate how decisive the battle was
            health_diff = abs(battle.get("strategy1_health", 0) - battle.get("strategy2_health", 0))
            
            summary = {
                "id": battle_id,
                "strategy1_id": battle.get("strategy1_id"),
                "strategy1_name": strategies.get(battle.get("strategy1_id", ""), {}).get("name", "Unknown"),
                "strategy2_id": battle.get("strategy2_id"),
                "strategy2_name": strategies.get(battle.get("strategy2_id", ""), {}).get("name", "Unknown"),
                "winner_id": battle.get("winner_id"),
                "winner_name": strategies.get(battle.get("winner_id", ""), {}).get("name", "Tie"),
                "strategy1_health": battle.get("strategy1_health"),
                "strategy2_health": battle.get("strategy2_health"),
                "health_diff": health_diff
            }
            battle_summary.append(summary)
        
        # Sort by health difference and take top 10
        battle_summary.sort(key=lambda x: x["health_diff"], reverse=True)
        battle_summary = battle_summary[:10]
        
        # Count strategies by template type
        template_counts = {}
        for strategy_id, strategy in strategies.items():
            template_type = strategy.get("template_type", "unknown")
            template_counts[template_type] = template_counts.get(template_type, 0) + 1
        
        return {
            "generation": generation,
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_count": len(strategies),
            "battle_count": len(battle_metrics),
            "template_counts": template_counts,
            "top_strategies": strategy_summary[:10],
            "all_strategies": strategy_summary,
            "notable_battles": battle_summary
        }
    
    def _prepare_final_report_data(
        self, 
        strategies: Dict[str, Dict[str, Any]], 
        battle_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for the final report."""
        # Group strategies by generation
        strategies_by_gen = {}
        max_generation = 0
        
        for strategy_id, strategy in strategies.items():
            generation = strategy.get("generation", 0)
            max_generation = max(max_generation, generation)
            
            if generation not in strategies_by_gen:
                strategies_by_gen[generation] = []
            
            strategies_by_gen[generation].append({
                "id": strategy_id,
                "name": strategy.get("name", strategy_id),
                "template_type": strategy.get("template_type", "unknown"),
                "win_rate": strategy.get("metrics", {}).get("win_rate", 0),
                "games_played": strategy.get("metrics", {}).get("games_played", 0),
                "avg_tower_health": strategy.get("metrics", {}).get("avg_tower_health", 0)
            })
        
        # Calculate performance trends by generation
        performance_by_gen = {}
        for gen, gen_strategies in strategies_by_gen.items():
            win_rates = [s["win_rate"] for s in gen_strategies if s["games_played"] > 0]
            health_values = [s["avg_tower_health"] for s in gen_strategies if s["games_played"] > 0]
            
            performance_by_gen[gen] = {
                "avg_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0,
                "max_win_rate": max(win_rates) if win_rates else 0,
                "avg_tower_health": sum(health_values) / len(health_values) if health_values else 0,
                "strategy_count": len(gen_strategies)
            }
        
        # Find most successful strategies across all generations
        all_strategies = []
        for gen, gen_strategies in strategies_by_gen.items():
            all_strategies.extend(gen_strategies)
        
        # Sort by win rate
        all_strategies.sort(key=lambda s: s["win_rate"], reverse=True)
        
        # Calculate performance by template type in final generation
        if max_generation in strategies_by_gen:
            final_gen_strategies = strategies_by_gen[max_generation]
            performance_by_template = {}
            
            for strategy in final_gen_strategies:
                template_type = strategy["template_type"]
                
                if template_type not in performance_by_template:
                    performance_by_template[template_type] = {
                        "strategies": [],
                        "avg_win_rate": 0,
                        "max_win_rate": 0
                    }
                
                performance_by_template[template_type]["strategies"].append(strategy)
            
            # Calculate averages
            for template_type, data in performance_by_template.items():
                win_rates = [s["win_rate"] for s in data["strategies"] if s["games_played"] > 0]
                
                if win_rates:
                    data["avg_win_rate"] = sum(win_rates) / len(win_rates)
                    data["max_win_rate"] = max(win_rates)
        else:
            performance_by_template = {}
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_strategies": len(strategies),
            "total_battles": len(battle_metrics),
            "max_generation": max_generation,
            "performance_by_gen": performance_by_gen,
            "top_strategies": all_strategies[:20],
            "performance_by_template": performance_by_template
        }
    
    def _generate_html_report(self, report_path: str, report_data: Dict[str, Any]) -> None:
        """Generate HTML report from report data."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Generation {report_data['generation']} Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .chart {{ margin: 20px 0; }}
        .card {{ box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); margin: 20px 0; padding: 20px; }}
    </style>
</head>
<body>
    <h1>Tower Defense RL - Generation {report_data['generation']} Report</h1>
    <p>Generated on: {report_data['timestamp']}</p>
    
    <div class="card">
        <h2>Summary</h2>
        <p>Total Strategies: {report_data['strategy_count']}</p>
        <p>Total Battles: {report_data['battle_count']}</p>
        <p>Template Distribution:</p>
        <ul>
        """
        
        for template, count in report_data["template_counts"].items():
            html += f"<li>{template}: {count}</li>\n"
        
        html += """
        </ul>
    </div>
    
    <div class="card">
        <h2>Visualizations</h2>
        <div class="chart">
            <img src="../../visualizations/win_rates_{report_data['generation']}.png" alt="Win Rates">
        </div>
        <div class="chart">
            <img src="../../../visualizations/fitness_components_{report_data['generation']}.png" alt="Fitness Components">
        </div>
    </div>
    
    <div class="card">
        <h2>Top Strategies</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Name</th>
                <th>Template</th>
                <th>Win Rate</th>
                <th>Games</th>
                <th>Avg Health</th>
                <th>Fitness</th>
            </tr>
        """
        
        for i, strategy in enumerate(report_data["top_strategies"]):
            html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{strategy['name']}</td>
                <td>{strategy['template_type']}</td>
                <td>{strategy['win_rate']:.2f}</td>
                <td>{strategy['games_played']}</td>
                <td>{strategy['avg_tower_health']:.1f}</td>
                <td>{strategy['fitness_score']:.4f}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
    
    <div class="card">
        <h2>Notable Battles</h2>
        <table>
            <tr>
                <th>Winner</th>
                <th>Strategy 1</th>
                <th>Health 1</th>
                <th>Strategy 2</th>
                <th>Health 2</th>
                <th>Health Diff</th>
            </tr>
        """
        
        for battle in report_data["notable_battles"]:
            html += f"""
            <tr>
                <td>{battle['winner_name']}</td>
                <td>{battle['strategy1_name']}</td>
                <td>{battle['strategy1_health']:.1f}</td>
                <td>{battle['strategy2_name']}</td>
                <td>{battle['strategy2_health']:.1f}</td>
                <td>{battle['health_diff']:.1f}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
</body>
</html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
    
    def _generate_final_html_report(self, report_path: str, report_data: Dict[str, Any]) -> None:
        """Generate final HTML report from report data."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tower Defense RL - Final Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .chart {{ margin: 20px 0; }}
        .card {{ box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); margin: 20px 0; padding: 20px; }}
    </style>
</head>
<body>
    <h1>Tower Defense RL - Final Report</h1>
    <p>Generated on: {report_data['timestamp']}</p>
    
    <div class="card">
        <h2>Summary</h2>
        <p>Total Strategies: {report_data['total_strategies']}</p>
        <p>Total Battles: {report_data['total_battles']}</p>
        <p>Generations: {report_data['max_generation'] + 1}</p>
    </div>
    
    <div class="card">
        <h2>Performance Evolution</h2>
        <div class="chart">
            <img src="../visualizations/performance_evolution.png" alt="Performance Evolution">
        </div>
        <table>
            <tr>
                <th>Generation</th>
                <th>Strategy Count</th>
                <th>Avg Win Rate</th>
                <th>Max Win Rate</th>
                <th>Avg Tower Health</th>
            </tr>
        """
        
        for gen in sorted(report_data["performance_by_gen"].keys()):
            data = report_data["performance_by_gen"][gen]
            html += f"""
            <tr>
                <td>{gen}</td>
                <td>{data['strategy_count']}</td>
                <td>{data['avg_win_rate']:.2f}</td>
                <td>{data['max_win_rate']:.2f}</td>
                <td>{data['avg_tower_health']:.1f}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
    
    <div class="card">
        <h2>Template Performance</h2>
        <div class="chart">
            <img src="../visualizations/archetype_comparison.png" alt="Template Comparison">
        </div>
        <table>
            <tr>
                <th>Template</th>
                <th>Strategy Count</th>
                <th>Avg Win Rate</th>
                <th>Max Win Rate</th>
            </tr>
        """
        
        for template, data in report_data["performance_by_template"].items():
            html += f"""
            <tr>
                <td>{template}</td>
                <td>{len(data['strategies'])}</td>
                <td>{data['avg_win_rate']:.2f}</td>
                <td>{data['max_win_rate']:.2f}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
    
    <div class="card">
        <h2>Top Strategies Overall</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Name</th>
                <th>Template</th>
                <th>Win Rate</th>
                <th>Games</th>
                <th>Avg Health</th>
            </tr>
        """
        
        for i, strategy in enumerate(report_data["top_strategies"]):
            html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{strategy['name']}</td>
                <td>{strategy['template_type']}</td>
                <td>{strategy['win_rate']:.2f}</td>
                <td>{strategy['games_played']}</td>
                <td>{strategy['avg_tower_health']:.1f}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
</body>
</html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
    
    def _generate_win_rate_chart(
        self, 
        generation: int, 
        strategies: Dict[str, Dict[str, Any]], 
        fitness_scores: Dict[str, float]
    ) -> str:
        """Generate win rate chart for a generation."""
        # Sort strategies by fitness score
        sorted_strategies = sorted(
            [(s_id, s) for s_id, s in strategies.items() if s.get("metrics", {}).get("games_played", 0) > 0],
            key=lambda x: fitness_scores.get(x[0], 0),
            reverse=True
        )
        
        # Extract data
        strategy_names = [s[1].get("name", s[0]) for s in sorted_strategies[:10]]
        win_rates = [s[1].get("metrics", {}).get("win_rate", 0) for s in sorted_strategies[:10]]
        template_types = [s[1].get("template_type", "unknown") for s in sorted_strategies[:10]]
        
        # Create color map for template types
        template_colors = {
            "counter_picker": "green",
            "resource_manager": "blue",
            "lane_adaptor": "orange",
            "phase_based": "purple",
            "direct_code": "red",
            "unknown": "gray"
        }
        
        colors = [template_colors.get(t, "gray") for t in template_types]
        
        # Create chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(strategy_names)), win_rates, color=colors)
        plt.xlabel('Strategy')
        plt.ylabel('Win Rate')
        plt.title(f'Top 10 Strategies by Win Rate - Generation {generation}')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
        plt.tight_layout()
        
        # Add legend
        legend_elements = []
        for template, color in template_colors.items():
            if template in template_types:
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=color, label=template))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save chart
        output_path = os.path.join(self.visualization_dir, f"win_rates_{generation}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _generate_fitness_components_chart(
        self, 
        generation: int, 
        strategies: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate fitness components chart for a generation."""
        # Sort strategies by win rate
        sorted_strategies = sorted(
            [(s_id, s) for s_id, s in strategies.items() if s.get("metrics", {}).get("games_played", 0) > 0],
            key=lambda x: x[1].get("metrics", {}).get("wilson_score", 0),
            reverse=True
        )
        
        # Extract data for top 10 strategies
        strategy_names = [s[1].get("name", s[0]) for s in sorted_strategies[:10]]
        wilson_scores = [s[1].get("metrics", {}).get("wilson_score", 0) for s in sorted_strategies[:10]]
        elite_performance = [s[1].get("metrics", {}).get("elite_performance", 0) for s in sorted_strategies[:10]]
        tower_health = [s[1].get("metrics", {}).get("avg_tower_health", 0) / 7032.0 for s in sorted_strategies[:10]]
        versatility = [s[1].get("metrics", {}).get("versatility", 0) for s in sorted_strategies[:10]]
        
        # Create chart
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(strategy_names))
        width = 0.2
        
        plt.bar(x - 1.5*width, wilson_scores, width, label='Wilson Score')
        plt.bar(x - 0.5*width, elite_performance, width, label='Elite Performance')
        plt.bar(x + 0.5*width, tower_health, width, label='Tower Health')
        plt.bar(x + 1.5*width, versatility, width, label='Versatility')
        
        plt.xlabel('Strategy')
        plt.ylabel('Score')
        plt.title(f'Fitness Components for Top 10 Strategies - Generation {generation}')
        plt.xticks(x, strategy_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(self.visualization_dir, f"fitness_components_{generation}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _generate_performance_evolution_chart(self, strategies: Dict[str, Dict[str, Any]]) -> str:
        """Generate performance evolution chart across generations."""
        # Group strategies by generation
        strategies_by_gen = {}
        
        for strategy_id, strategy in strategies.items():
            generation = strategy.get("generation", 0)
            
            if generation not in strategies_by_gen:
                strategies_by_gen[generation] = []
            
            strategies_by_gen[generation].append(strategy)
        
        # Calculate performance metrics by generation
        generations = sorted(strategies_by_gen.keys())
        avg_win_rates = []
        max_win_rates = []
        avg_health = []
        
        for gen in generations:
            gen_strategies = strategies_by_gen[gen]
            win_rates = [s.get("metrics", {}).get("win_rate", 0) for s in gen_strategies 
                        if s.get("metrics", {}).get("games_played", 0) > 0]
            health_values = [s.get("metrics", {}).get("avg_tower_health", 0) for s in gen_strategies 
                           if s.get("metrics", {}).get("games_played", 0) > 0]
            
            avg_win_rates.append(sum(win_rates) / len(win_rates) if win_rates else 0)
            max_win_rates.append(max(win_rates) if win_rates else 0)
            avg_health.append(sum(health_values) / len(health_values) if health_values else 0)
        
        # Create chart
        plt.figure(figsize=(12, 6))
        
        plt.plot(generations, avg_win_rates, 'b-', label='Average Win Rate')
        plt.plot(generations, max_win_rates, 'g-', label='Max Win Rate')
        
        # Normalize avg_health for comparison
        normalized_health = [h / 7032.0 for h in avg_health]
        plt.plot(generations, normalized_health, 'r-', label='Avg Tower Health (normalized)')
        
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('Performance Evolution Across Generations')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(self.visualization_dir, "performance_evolution.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _generate_archetype_comparison_chart(self, strategies: Dict[str, Dict[str, Any]]) -> str:
        """Generate a comparison chart of different strategy archetypes."""
        # Group strategies by template type and generation
        template_groups = {}
        
        for strategy_id, strategy in strategies.items():
            template_type = strategy.get("template_type", "unknown")
            generation = strategy.get("generation", 0)
            
            if template_type not in template_groups:
                template_groups[template_type] = {}
            
            if generation not in template_groups[template_type]:
                template_groups[template_type][generation] = []
            
            template_groups[template_type][generation].append(strategy)
        
        # Calculate average win rate by template type and generation
        templates = sorted(template_groups.keys())
        generations = sorted(set(gen for template in template_groups.values() for gen in template.keys()))
        
        template_win_rates = {}
        
        for template in templates:
            template_win_rates[template] = []
            
            for gen in generations:
                if gen in template_groups[template]:
                    gen_strategies = template_groups[template][gen]
                    win_rates = [s.get("metrics", {}).get("win_rate", 0) for s in gen_strategies 
                               if s.get("metrics", {}).get("games_played", 0) > 0]
                    
                    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
                else:
                    avg_win_rate = 0
                
                template_win_rates[template].append(avg_win_rate)
        
        # Create chart
        plt.figure(figsize=(12, 6))
        
        for template in templates:
            plt.plot(generations, template_win_rates[template], marker='o', label=template)
        
        plt.xlabel('Generation')
        plt.ylabel('Average Win Rate')
        plt.title('Archetype Performance Comparison Across Generations')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(self.visualization_dir, "archetype_comparison.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path