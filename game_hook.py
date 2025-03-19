import os
import json
import sys
import inspect
import pygame
from pathlib import Path
from datetime import datetime

class GameHook:
    """
    Hooks into the game to extract detailed metrics during gameplay.
    Modifies game files temporarily to add instrumentation.
    """
    
    def __init__(self, game_dir="game", backup_suffix=".bak", metrics_dir="training_data/game_metrics"):
        """
        Initialize the game hook.
        
        Args:
            game_dir: Directory containing game files
            backup_suffix: Suffix for backup files
            metrics_dir: Directory to save extracted metrics
        """
        self.game_dir = game_dir
        self.backup_suffix = backup_suffix
        self.metrics_dir = metrics_dir
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)
        
        # Files to instrument
        self.files_to_hook = {
            "game.py": self._hook_game_file,
            "scripts/dataflow.py": self._hook_dataflow_file,
            "scripts/decoration.py": self._hook_decoration_file
        }
        
        # Keep track of backups made
        self.backups = []
    
    def install_hooks(self):
        """Install all hooks into game files."""
        print("Installing game hooks...")
        
        for filepath, hook_function in self.files_to_hook.items():
            full_path = os.path.join(self.game_dir, filepath)
            if os.path.exists(full_path):
                # Create backup
                backup_path = full_path + self.backup_suffix
                if not os.path.exists(backup_path):
                    print(f"Backing up {filepath}")
                    with open(full_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    
                    self.backups.append(full_path)
                
                # Apply hook
                print(f"Instrumenting {filepath}")
                hook_function(full_path)
            else:
                print(f"Warning: File not found: {full_path}")
        
        print("Game hooks installed successfully")
    
    def restore_backups(self):
        """Restore original game files from backups."""
        print("Restoring original game files...")
        
        for filepath in self.backups:
            backup_path = filepath + self.backup_suffix
            if os.path.exists(backup_path):
                print(f"Restoring {os.path.basename(filepath)}")
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Keep backups for safety, uncomment to remove them
                # os.remove(backup_path)
        
        self.backups = []
        print("Original game files restored")
    
    def _hook_game_file(self, filepath):
        """Add hooks to game.py to extract game state and metrics."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add import for our metrics collector
        import_line = "import random\nfrom scripts.game_config import *\n"
        metrics_import = "import random\nimport json\nimport os\nfrom datetime import datetime\nfrom scripts.game_config import *\n"
        content = content.replace(import_line, metrics_import)
        
        # Add metrics collection methods to Game class - using triple single quotes for inner docstring
        class_def = "class Game:\n"
        metrics_methods = '''class Game:
    def collect_metrics(self):
        """Collect detailed game metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "game_time": self.game_counter,
            "tower1": {
                "name": self.team_name1,
                "health": self.tower1.health,
                "elixir": self.tower1.total_elixir,
                "troop_count": len(self.tower1.myTroops)
            },
            "tower2": {
                "name": self.team_name2,
                "health": self.tower2.health,
                "elixir": self.tower2.total_elixir,
                "troop_count": len(self.tower2.myTroops)
            },
            "troops1": [],
            "troops2": []
        }
        
        # Collect troop data
        for troop in self.tower1.myTroops:
            metrics["troops1"].append({
                "name": troop.name,
                "health": troop.health,
                "position": troop.position,
                "type": troop.type
            })
            
        for troop in self.tower2.myTroops:
            metrics["troops2"].append({
                "name": troop.name,
                "health": troop.health,
                "position": troop.position,
                "type": troop.type
            })
        
        # Save metrics to file
        metrics_dir = os.path.join("training_data", "game_metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_file = os.path.join(metrics_dir, f"game_{self.game_counter}.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
'''
        content = content.replace(class_def, metrics_methods)
        
        # Add metrics collection to render_game_screen
        render_method = "    def render_game_screen(self):"
        metrics_render = """    def render_game_screen(self):
        # Collect metrics every 30 frames
        if self.game_counter % 30 == 0 and GAME_END_TIME > self.game_counter >= GAME_START_TIME:
            self.collect_metrics()
            
"""
        content = content.replace(render_method, metrics_render)
        
        # Write modified content back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _hook_dataflow_file(self, filepath):
        """Add hooks to dataflow.py to extract deployment and attack data."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add deployment tracking
        deployment_method = "    def deployment(self):"
        deployment_hook = """    def deployment(self):
        # Record deployment data
        deploy_metrics_file = os.path.join("training_data", "game_metrics", "deployments.json")
        
        try:
            if os.path.exists(deploy_metrics_file):
                with open(deploy_metrics_file, 'r') as f:
                    deploy_data = json.load(f)
            else:
                deploy_data = {"team1": [], "team2": []}
                
            # Capture pre-deployment state
            pre_deploy = {
                "game_time": self.game_counter,
                "tower1_elixir": self.tower1.total_elixir,
                "tower2_elixir": self.tower2.total_elixir
            }
"""
        content = content.replace(deployment_method, deployment_hook)
        
        # Add post-deployment tracking
        post_deployment = "        self.data_provided1 = {}\n        self.data_provided2 = {}"
        post_hook = """        # Record post-deployment data
        post_deploy = {
            "game_time": self.game_counter,
            "tower1_elixir": self.tower1.total_elixir,
            "tower2_elixir": self.tower2.total_elixir,
            "team1_deployed": [t[0] for t in troops1_list],
            "team2_deployed": [t[0] for t in troops2_list]
        }
        
        # Save combined data
        deploy_entry = {
            "pre": pre_deploy,
            "post": post_deploy,
            "team1_signal": team_signal1,
            "team2_signal": team_signal2
        }
        
        deploy_data["team1"].append({
            "time": self.game_counter,
            "troops": [t[0] for t in troops1_list],
            "positions": [t[1] for t in troops1_list],
            "elixir_before": pre_deploy["tower1_elixir"],
            "elixir_after": post_deploy["tower1_elixir"]
        })
        
        deploy_data["team2"].append({
            "time": self.game_counter,
            "troops": [t[0] for t in troops2_list],
            "positions": [t[1] for t in troops2_list],
            "elixir_before": pre_deploy["tower2_elixir"],
            "elixir_after": post_deploy["tower2_elixir"]
        })
            
        with open(deploy_metrics_file, 'w') as f:
            json.dump(deploy_data, f, indent=2)
    
        self.data_provided1 = {}
        self.data_provided2 = {}"""
        
        content = content.replace(post_deployment, post_hook)
        
        # Write modified content back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _hook_decoration_file(self, filepath):
        """Add hooks to decoration.py to record game outcomes."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add outcome recording
        check_game_end = "    def check_game_end(self):"
        outcome_hook = """    def check_game_end(self):
        # Track when game is about to end
        game_ending = False
"""
        content = content.replace(check_game_end, outcome_hook)
        
        # Add final outcome recording - fixing the pattern and replacement
        winner_check = "if self.winner == self.team_name2"
        outcome_recording = """
        # Record game outcome
        game_ending = True
        outcome_file = os.path.join("training_data", "game_metrics", "outcome.json")
        
        outcome = {
            "winner": self.winner,
            "team1_name": self.team_name1,
            "team2_name": self.team_name2,
            "team1_health": self.tower1.health,
            "team2_health": self.tower2.health,
            "game_duration": self.game_counter,
            "message": self.message
        }
        
        try:
            with open(outcome_file, 'w') as f:
                json.dump(outcome, f, indent=2)
        except Exception as e:
            print(f"Error saving outcome: {e}")
            
        if self.winner == self.team_name2"""
        
        content = content.replace(winner_check, outcome_recording)
        
        # Write modified content back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def extract_metrics(self):
        """
        Process and extract insights from collected game metrics.
        Returns a dictionary with aggregated metrics.
        """
        print("Extracting metrics from game data...")
        
        # Initialize results dictionary
        results = {
            "deployment_patterns": {
                "team1": {},
                "team2": {}
            },
            "troop_effectiveness": {},
            "elixir_efficiency": {},
            "position_heatmap": {}
        }
        
        # Process deployment data
        deploy_file = os.path.join(self.metrics_dir, "deployments.json")
        if os.path.exists(deploy_file):
            try:
                with open(deploy_file, 'r') as f:
                    deploy_data = json.load(f)
                
                # Analyze deployment patterns
                for team, deployments in deploy_data.items():
                    troop_counts = {}
                    position_data = []
                    elixir_spent = 0
                    
                    for deploy in deployments:
                        for troop in deploy.get("troops", []):
                            if troop not in troop_counts:
                                troop_counts[troop] = 0
                            troop_counts[troop] += 1
                        
                        position_data.extend(deploy.get("positions", []))
                        elixir_spent += deploy.get("elixir_before", 0) - deploy.get("elixir_after", 0)
                    
                    if team == "team1":
                        results["deployment_patterns"]["team1"] = {
                            "troop_frequency": troop_counts,
                            "total_deployments": len(deployments),
                            "elixir_spent": elixir_spent
                        }
                    else:
                        results["deployment_patterns"]["team2"] = {
                            "troop_frequency": troop_counts,
                            "total_deployments": len(deployments),
                            "elixir_spent": elixir_spent
                        }
                
            except Exception as e:
                print(f"Error processing deployment data: {e}")
        
        # Process game metrics files
        metrics_files = [f for f in os.listdir(self.metrics_dir) if f.startswith("game_") and f.endswith(".json")]
        
        troop_stats = {}
        
        for file in metrics_files:
            try:
                with open(os.path.join(self.metrics_dir, file), 'r') as f:
                    metrics = json.load(f)
                
                # Process troop effectiveness
                for team_key, team_troops in [("troops1", "team1"), ("troops2", "team2")]:
                    for troop in metrics.get(team_key, []):
                        name = troop.get("name")
                        if name not in troop_stats:
                            troop_stats[name] = {
                                "count": 0,
                                "avg_health": 0,
                                "positions": []
                            }
                        
                        troop_stats[name]["count"] += 1
                        troop_stats[name]["avg_health"] = (
                            (troop_stats[name]["avg_health"] * (troop_stats[name]["count"] - 1) +
                             troop.get("health", 0)) / troop_stats[name]["count"]
                        )
                        troop_stats[name]["positions"].append(troop.get("position"))
                
            except Exception as e:
                print(f"Error processing metrics file {file}: {e}")
        
        results["troop_effectiveness"] = troop_stats
        
        # Process outcome file
        outcome_file = os.path.join(self.metrics_dir, "outcome.json")
        if os.path.exists(outcome_file):
            try:
                with open(outcome_file, 'r') as f:
                    outcome = json.load(f)
                    
                results["outcome"] = outcome
                
                # Calculate elixir efficiency
                team1_elixir = results["deployment_patterns"]["team1"].get("elixir_spent", 0)
                team2_elixir = results["deployment_patterns"]["team2"].get("elixir_spent", 0)
                
                if team1_elixir > 0:
                    results["elixir_efficiency"]["team1"] = (
                        (10000 - outcome.get("team2_health", 0)) / team1_elixir
                    )
                
                if team2_elixir > 0:
                    results["elixir_efficiency"]["team2"] = (
                        (10000 - outcome.get("team1_health", 0)) / team2_elixir
                    )
                
            except Exception as e:
                print(f"Error processing outcome data: {e}")
        
        # Create position heatmap data
        for troop_name, stats in troop_stats.items():
            positions = stats.get("positions", [])
            if positions:
                # Create a simplified 10x10 grid for positions
                heatmap = [[0 for _ in range(10)] for _ in range(10)]
                
                for pos in positions:
                    if pos and len(pos) == 2:
                        x, y = pos
                        # Normalize to 0-9 grid
                        x_idx = min(9, max(0, int((x + 60) / 12)))
                        y_idx = min(9, max(0, int(y / 6)))
                        
                        heatmap[y_idx][x_idx] += 1
                
                results["position_heatmap"][troop_name] = heatmap
        
        # Save processed metrics
        processed_file = os.path.join(self.metrics_dir, "processed_metrics.json")
        with open(processed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Metrics extracted and saved to {processed_file}")
        return results


if __name__ == "__main__":
    # Simple test
    hook = GameHook()
    
    try:
        hook.install_hooks()
        print("Game hooks installed. Run a game to collect metrics.")
        print("After running the game, use hook.extract_metrics() to process collected data.")
        
        # Uncomment to restore original files
        # hook.restore_backups()
        
    except Exception as e:
        print(f"Error: {e}")
        # Try to restore backups on error
        hook.restore_backups()