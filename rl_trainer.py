import os
import json
import random
import uuid
import importlib.util
import sys
import time
import shutil
from pathlib import Path

class RLTrainer:
    def __init__(self, output_dir="training_data"):
        """Initialize the RL training system"""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.metrics_file = os.path.join(output_dir, "battle_metrics.json")
        self.strategies_file = os.path.join(output_dir, "strategies.json")
        self.load_strategies()
        
    def load_strategies(self):
        """Load existing strategies or create initial ones"""
        if os.path.exists(self.strategies_file):
            with open(self.strategies_file, 'r') as f:
                self.strategies = json.load(f)
            print(f"Loaded {len(self.strategies)} existing strategies")
        else:
            # Create initial strategy templates
            self.strategies = {
                "baseline": self.create_initial_strategy("baseline"),
                "aggressive": self.create_initial_strategy("aggressive"),
                "defensive": self.create_initial_strategy("defensive"),
                "balanced": self.create_initial_strategy("balanced"),
                "pratyaksh": self.create_initial_strategy("pratyaksh"),  # Add the new strategy
            }
            self.save_strategies()
            print("Created initial strategies")
    
    def create_initial_strategy(self, strategy_type):
        """Create an initial strategy based on type"""
        troop_selection = [
            "dragon", "wizard", "valkyrie", "musketeer",
            "knight", "archer", "minion", "barbarian"
        ]
        
        # Default values
        strategy = {
            "id": f"{strategy_type}_{str(uuid.uuid4())[:8]}",
            "name": strategy_type,
            "troop_selection": troop_selection,
            "lane_preference": "right",
            "elixir_thresholds": {
                "min_deploy": 4,
                "save_threshold": 7,
                "emergency_threshold": 2
            },
            "position_settings": {
                "x_range": [-20, 20],
                "y_default": 45,
                "defensive_y": 20
            },
            "defensive_trigger": 0.7,
            "counter_weights": {
                "air_vs_ground": 1.2,
                "splash_vs_group": 1.5,
                "tank_priority": 1.3
            },
            "metrics": {
                "games_played": 0,
                "wins": 0,
                "losses": 0,
                "avg_tower_health": 0,
                "win_rate": 0
            }
        }
        
        # Customize based on strategy type
        if strategy_type == "aggressive":
            strategy["lane_preference"] = "right"
            strategy["elixir_thresholds"]["min_deploy"] = 3
            strategy["position_settings"]["y_default"] = 49
            strategy["defensive_trigger"] = 0.4
            strategy["troop_priority"] = ["dragon", "prince", "wizard", "musketeer", "valkyrie", "barbarian", "knight", "archer", "minion"]
            
        elif strategy_type == "defensive":
            strategy["lane_preference"] = "split"
            strategy["elixir_thresholds"]["min_deploy"] = 5
            strategy["position_settings"]["y_default"] = 30
            strategy["defensive_trigger"] = 0.9
            strategy["troop_priority"] = ["wizard", "musketeer", "valkyrie", "knight", "dragon", "archer", "barbarian", "minion"]
            
        elif strategy_type == "balanced":
            strategy["lane_preference"] = "center"
            strategy["elixir_thresholds"]["min_deploy"] = 4
            strategy["position_settings"]["y_default"] = 40
            strategy["defensive_trigger"] = 0.6
            strategy["troop_priority"] = ["dragon", "wizard", "valkyrie", "knight", "musketeer", "archer", "barbarian", "minion"]
            
        elif strategy_type == "pratyaksh":
            # Add settings for the Pratyaksh strategy
            strategy["lane_preference"] = "adaptive"
            strategy["elixir_thresholds"]["min_deploy"] = 3
            strategy["position_settings"]["y_default"] = 40
            strategy["position_settings"]["x_range"] = [-25, 25]
            strategy["defensive_trigger"] = 0.65
            strategy["counter_weights"]["air_vs_ground"] = 1.5
            strategy["troop_priority"] = ["dragon", "wizard", "musketeer", "valkyrie", "archer", "skeleton", "barbarian", "minion"]
            # Add additional parameters for counter-strategy logic
            strategy["counter_strategy"] = {
                "analyze_opponent": True,
                "air_counters": ["archer", "musketeer", "wizard"],
                "ground_counters": ["dragon", "minion", "valkyrie"],
                "counter_bonus": 3.0
            }
        else:  # baseline
            strategy["lane_preference"] = "right"
            strategy["elixir_thresholds"]["min_deploy"] = 4
            strategy["position_settings"]["y_default"] = 45
            strategy["defensive_trigger"] = 0.5
            strategy["troop_priority"] = ["wizard", "dragon", "musketeer", "valkyrie", "knight", "archer", "barbarian", "minion"]
            
        return strategy
        
    def save_strategies(self):
        """Save all strategies to the strategies file"""
        with open(self.strategies_file, 'w') as f:
            json.dump(self.strategies, f, indent=2)
        print(f"Saved {len(self.strategies)} strategies to {self.strategies_file}")
    
    def generate_team_file(self, strategy, team_name, output_path=None):
        """Generate a team Python file from strategy parameters"""
        from strategy_generator import StrategyGenerator
        generator = StrategyGenerator()
        
        # Generate code
        code = generator.generate_code(strategy)
        
        # Save to file if path specified
        if output_path:
            with open(output_path, 'w') as f:
                f.write(code)
            return output_path
            
        # Otherwise save to temporary file
        temp_dir = os.path.join(self.output_dir, "temp")
        Path(temp_dir).mkdir(exist_ok=True, parents=True)
        file_path = os.path.join(temp_dir, f"{team_name}_{str(uuid.uuid4())[:8]}.py")
        with open(file_path, 'w') as f:
            f.write(code)
            
        return file_path
    
    def run_battle(self, strategy1, strategy2):
        """Run a battle between two strategies and return the result"""
        # Generate team files
        team1_file = self.generate_team_file(strategy1, "team1")
        team2_file = self.generate_team_file(strategy2, "team2")
        
        # Prepare to run the game
        # Modify game configuration to use these teams
        self.update_game_config(team1_file, team2_file)
        
        # Run the game and collect results
        result = self.run_game()
        
        return result
    
    def update_game_config(self, team1_path, team2_path):
        """Update the game config to use our generated teams"""
        # Extract just the filename without extension
        team1_name = os.path.basename(team1_path).split('.')[0]
        team2_name = os.path.basename(team2_path).split('.')[0]
        
        # Copy files to teams directory
        shutil.copy(team1_path, "game/teams/a.py")
        shutil.copy(team2_path, "game/teams/b.py")
        
        print(f"Updated game configuration with teams: {team1_name} vs {team2_name}")
    
    def run_game(self):
        """Run the game and return the result"""
        # In a real implementation, this would run the game and extract results
        # For now, we'll simulate the game result
        result = {
            "winner": "team1" if random.random() > 0.5 else "team2",
            "team1_health": random.randint(0, 1000),
            "team2_health": random.randint(0, 1000),
            "duration": random.randint(300, 1800)
        }
        
        print(f"Game result: Winner={result['winner']}, " +
              f"Team1 Health={result['team1_health']}, " +
              f"Team2 Health={result['team2_health']}")
        
        return result
    
    def run_training_cycle(self, generations=5, battles_per_gen=3):
        """Run a complete training cycle with multiple generations"""
        for gen in range(generations):
            print(f"\n=== Generation {gen+1}/{generations} ===")
            
            # Select strategies to battle
            battle_pairs = self.select_battle_pairs(battles_per_gen)
            
            # Run battles
            results = []
            for strategy1_id, strategy2_id in battle_pairs:
                print(f"\nBattle: {strategy1_id} vs {strategy2_id}")
                result = self.run_battle(
                    self.strategies[strategy1_id], 
                    self.strategies[strategy2_id]
                )
                results.append((strategy1_id, strategy2_id, result))
                
                # Add brief pause between battles
                time.sleep(1)
            
            # Update strategy performance metrics
            self.update_strategy_metrics(results)
            
            # Evolve new strategies
            self.evolve_strategies()
            
            # Save progress
            self.save_strategies()
            
            # Display current best strategies
            self.display_top_strategies(5)
    
    def select_battle_pairs(self, count):
        """Select pairs of strategies to battle each other"""
        strategies = list(self.strategies.keys())
        pairs = []
        
        # Ensure each strategy fights at least once
        for strategy_id in strategies[:count]:
            # Find an opponent (not itself)
            opponents = [s for s in strategies if s != strategy_id]
            if opponents:
                opponent_id = random.choice(opponents)
                pairs.append((strategy_id, opponent_id))
        
        # Add additional random pairs if needed
        while len(pairs) < count:
            s1 = random.choice(strategies)
            s2 = random.choice([s for s in strategies if s != s1])
            pairs.append((s1, s2))
        
        return pairs[:count]  # Limit to requested count
    
    def update_strategy_metrics(self, results):
        """Update metrics for each strategy based on battle results"""
        for strategy1_id, strategy2_id, result in results:
            # Update games played
            self.strategies[strategy1_id]["metrics"]["games_played"] += 1
            self.strategies[strategy2_id]["metrics"]["games_played"] += 1
            
            # Update wins/losses
            if result["winner"] == "team1":
                self.strategies[strategy1_id]["metrics"]["wins"] += 1
                self.strategies[strategy2_id]["metrics"]["losses"] += 1
                # Update tower health
                self.strategies[strategy1_id]["metrics"]["avg_tower_health"] = (
                    (self.strategies[strategy1_id]["metrics"]["avg_tower_health"] * 
                     (self.strategies[strategy1_id]["metrics"]["games_played"] - 1) + 
                     result["team1_health"]) / 
                    self.strategies[strategy1_id]["metrics"]["games_played"]
                )
            else:
                self.strategies[strategy2_id]["metrics"]["wins"] += 1
                self.strategies[strategy1_id]["metrics"]["losses"] += 1
                # Update tower health
                self.strategies[strategy2_id]["metrics"]["avg_tower_health"] = (
                    (self.strategies[strategy2_id]["metrics"]["avg_tower_health"] * 
                     (self.strategies[strategy2_id]["metrics"]["games_played"] - 1) + 
                     result["team2_health"]) / 
                    self.strategies[strategy2_id]["metrics"]["games_played"]
                )
            
            # Update win rates
            for strategy_id in [strategy1_id, strategy2_id]:
                metrics = self.strategies[strategy_id]["metrics"]
                if metrics["games_played"] > 0:
                    metrics["win_rate"] = metrics["wins"] / metrics["games_played"]
    
    def evolve_strategies(self):
        """Create new strategies through evolution"""
        # This will be implemented in the evolve.py module later
        # For now, we'll just create a simple random variation
        
        # Find the best strategy
        best_strategy_id = self.find_best_strategy()
        if not best_strategy_id:
            return
            
        best_strategy = self.strategies[best_strategy_id]
        
        # Create a mutated version
        new_strategy = self.mutate_strategy(best_strategy)
        
        # Add to strategies
        self.strategies[new_strategy["id"]] = new_strategy
        print(f"Created new evolved strategy: {new_strategy['id']}")
    
    def mutate_strategy(self, strategy):
        """Create a mutated copy of a strategy"""
        import copy
        new_strategy = copy.deepcopy(strategy)
        
        # Generate new ID and name
        new_strategy["id"] = f"evolved_{str(uuid.uuid4())[:8]}"
        new_strategy["name"] = f"evolved_{strategy['name']}"
        
        # Reset metrics
        new_strategy["metrics"] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "avg_tower_health": 0,
            "win_rate": 0
        }
        
        # Apply some simple mutations
        # Mutate elixir thresholds
        for key in new_strategy["elixir_thresholds"]:
            new_strategy["elixir_thresholds"][key] = max(
                1, min(10, new_strategy["elixir_thresholds"][key] + 
                      random.uniform(-1, 1))
            )
        
        # Mutate position settings
        new_strategy["position_settings"]["x_range"][0] += random.randint(-5, 5)
        new_strategy["position_settings"]["x_range"][1] += random.randint(-5, 5)
        new_strategy["position_settings"]["y_default"] = max(
            0, min(49, new_strategy["position_settings"]["y_default"] + 
                  random.randint(-10, 10))
        )
        
        # Mutate defensive trigger
        new_strategy["defensive_trigger"] = max(
            0.1, min(0.9, new_strategy["defensive_trigger"] + 
                    random.uniform(-0.2, 0.2))
        )
        
        # Potentially shuffle troop priority
        if "troop_priority" in new_strategy and random.random() > 0.5:
            # Shuffle the first few troops in the priority list
            cutoff = random.randint(1, 3)
            start = new_strategy["troop_priority"][:cutoff]
            rest = new_strategy["troop_priority"][cutoff:]
            random.shuffle(start)
            new_strategy["troop_priority"] = start + rest
            
        return new_strategy
    
    def find_best_strategy(self):
        """Find the best strategy using Wilson score confidence interval"""
        try:
            # Try to use the improved strategy selection
            from improved_strategy_selection import select_best_strategies
            
            # Load battle metrics
            battle_metrics = []
            metrics_file = os.path.join(self.output_dir, "battle_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    battle_metrics = json.load(f)
            
            # Get sorted strategies using Wilson score
            sorted_strategies = select_best_strategies(
                battle_metrics,
                min_games=3,  # Require at least 3 games
                confidence=0.95
            )
            
            if sorted_strategies:
                best_strategy_id = sorted_strategies[0][0]
                stats = sorted_strategies[0][1]
                print(f"Best strategy: {best_strategy_id} with Wilson score: {stats['wilson_score']:.2f} ({stats['wins']}/{stats['games']} wins)")
                return best_strategy_id
        
        except ImportError as e:
            print(f"Warning: Could not use improved strategy selection: {e}")
            print("Falling back to basic win rate method.")
        
        # Fallback to simple win rate if Wilson score method fails
        best_win_rate = -1
        best_games_played = 0
        best_strategy_id = None
        
        for strategy_id, strategy in self.strategies.items():
            metrics = strategy["metrics"]
            games_played = metrics["games_played"]
            
            # Only consider strategies with at least 3 games
            if games_played >= 3:
                win_rate = metrics["win_rate"]
                
                # Prefer strategies with higher win rates
                # If win rates are equal, prefer the one with more games played
                if win_rate > best_win_rate or (win_rate == best_win_rate and games_played > best_games_played):
                    best_win_rate = win_rate
                    best_games_played = games_played
                    best_strategy_id = strategy_id
        
        return best_strategy_id
    
    def display_top_strategies(self, n=5):
        """Display the top N strategies by win rate"""
        try:
            # Try to use the improved strategy selection
            from improved_strategy_selection import select_best_strategies
            
            # Load battle metrics
            battle_metrics = []
            metrics_file = os.path.join(self.output_dir, "battle_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    battle_metrics = json.load(f)
            
            # Get sorted strategies using Wilson score
            sorted_strategies = select_best_strategies(
                battle_metrics,
                min_games=3,  # Require at least 3 games
                confidence=0.95
            )
            
            print("\nTop Strategies (Wilson Score):")
            for i, (strategy_id, stats) in enumerate(sorted_strategies[:n]):
                print(f"{i+1}. {strategy_id}: Win Rate={stats['win_rate']:.2f} ({stats['wins']}/{stats['games']} games), Wilson Score={stats['wilson_score']:.2f}")
            
        except ImportError:
            # Fallback to simple win rate
            # Sort strategies by win rate
            sorted_strategies = sorted(
                [(s_id, s["metrics"]["win_rate"], s["metrics"]["wins"], s["metrics"]["games_played"]) 
                for s_id, s in self.strategies.items() if s["metrics"]["games_played"] >= 3],
                key=lambda x: x[1],
                reverse=True
            )
            
            print("\nTop Strategies (Win Rate):")
            for i, (strategy_id, win_rate, wins, games_played) in enumerate(sorted_strategies[:n]):
                print(f"{i+1}. {strategy_id}: Win Rate={win_rate:.2f} ({wins}/{games_played} games)")
    
    def generate_final_submission(self):
        """Generate the final submission file"""
        # Find the best strategy
        best_strategy_id = self.find_best_strategy()
        
        if not best_strategy_id:
            print("No strategies found with enough games played. Cannot generate submission.")
            return
            
        best_strategy = self.strategies.get(best_strategy_id)
        if not best_strategy:
            print(f"Error: Strategy ID {best_strategy_id} not found in strategies.")
            # Try to find the strategy in the team files
            for gen_folder in sorted(os.listdir(os.path.join(self.output_dir, "teams")), reverse=True):
                if gen_folder.startswith("gen_"):
                    potential_path = os.path.join(self.output_dir, "teams", gen_folder, f"{best_strategy_id}.py")
                    if os.path.exists(potential_path):
                        print(f"Found best strategy file at: {potential_path}")
                        # Copy file directly to submission
                        with open(potential_path, 'r') as f:
                            code = f.read()
                        submission_file = "submission.py"
                        with open(submission_file, 'w') as f:
                            f.write(code)
                        print(f"Final submission generated at: {submission_file}")
                        return
            return
        
        # Generate the final submission file
        submission_file = "submission.py"
        self.generate_team_file(best_strategy, "submission", submission_file)
        
        # Also copy to teams directory
        teams_dir = "teams"
        if not os.path.exists(teams_dir):
            os.makedirs(teams_dir)
        shutil.copy(submission_file, os.path.join(teams_dir, "submission.py"))
        
        print(f"Final submission generated at: {submission_file}")
        print(f"Copy also saved to: teams/submission.py")
        print(f"Best strategy: {best_strategy_id} with win rate: {best_strategy['metrics']['win_rate']:.2f}")

if __name__ == "__main__":
    # Simple test run
    trainer = RLTrainer()
    trainer.run_training_cycle(generations=2, battles_per_gen=2)
    trainer.generate_final_submission()