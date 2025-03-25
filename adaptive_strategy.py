"""
Adaptive strategy generation for tower defense.
Contains all logic for analyzing opponents and creating counter-strategies.
"""

import random
import copy
import uuid
from collections import Counter, defaultdict

class AdaptiveStrategyManager:
    """
    Handles adaptive strategy creation and opponent analysis.
    All-in-one class to minimize changes to existing code.
    """
    
    def __init__(self, battle_metrics=None):
        """
        Initialize the adaptive strategy manager.
        
        Args:
            battle_metrics: Optional list of battle metrics for analysis
        """
        self.battle_metrics = battle_metrics or []
        
        # Troop type classifications
        self.troop_types = {
            "air": ["dragon", "minion"],
            "ground": ["knight", "archer", "valkyrie", "wizard", "musketeer", "prince", "giant", "skeleton", "bomber"],
            "splash": ["wizard", "valkyrie", "bomber"],
            "single": ["knight", "archer", "musketeer", "prince", "giant", "skeleton", "dragon", "minion"],
            "tank": ["knight", "giant", "prince", "valkyrie"],
            "dps": ["archer", "wizard", "musketeer", "dragon", "minion", "skeleton", "bomber"]
        }
        
        # Troop counter mappings
        self.troop_counters = {
            "dragon": ["archer", "musketeer", "wizard"],
            "minion": ["archer", "musketeer", "wizard"],
            "wizard": ["knight", "giant", "prince"],
            "musketeer": ["knight", "valkyrie", "prince"],
            "valkyrie": ["minion", "dragon", "skeleton"],
            "knight": ["skeleton", "minion", "dragon"],
            "archer": ["knight", "giant", "valkyrie"],
            "prince": ["skeleton", "archer", "wizard"],
            "giant": ["skeleton", "minion", "valkyrie"],
            "skeleton": ["wizard", "valkyrie", "bomber"]
        }
        
        # Base strategies for adaptation
        self.base_strategies = {
            "anti_air": {
                "troop_priority": ["archer", "musketeer", "wizard", "knight", "valkyrie", "prince", "skeleton"],
                "counter_weights": {"air_vs_ground": 3.0, "splash_vs_group": 1.0, "tank_priority": 1.0},
                "lane_preference": "adaptive"
            },
            "anti_splash": {
                "troop_priority": ["dragon", "minion", "prince", "knight", "musketeer", "archer", "skeleton"], 
                "counter_weights": {"air_vs_ground": 1.0, "splash_vs_group": 0.5, "tank_priority": 2.0},
                "lane_preference": "split"
            },
            "anti_tank": {
                "troop_priority": ["skeleton", "minion", "valkyrie", "wizard", "musketeer", "dragon", "archer"],
                "counter_weights": {"air_vs_ground": 1.5, "splash_vs_group": 2.0, "tank_priority": 0.7},
                "lane_preference": "center"
            }
        }
    
    def set_battle_metrics(self, metrics):
        """Set the battle metrics for analysis."""
        self.battle_metrics = metrics
    
    def analyze_strategy(self, strategy_id):
        """
        Analyze a strategy based on its battle history.
        
        Args:
            strategy_id: ID of the strategy to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        # Find all battles involving this strategy
        battles = []
        for battle in self.battle_metrics:
            team1_file = battle.get("team1_file", "")
            team2_file = battle.get("team2_file", "")
            
            team1_id = team1_file
            if team1_file.endswith('.py'):
                team1_id = team1_file[:-3]
                
            team2_id = team2_file
            if team2_file.endswith('.py'):
                team2_id = team2_file[:-3]
            
            if team1_id == strategy_id or team2_id == strategy_id:
                battles.append(battle)
        
        if not battles:
            return None
        
        # Analyze troop usage and lane preference
        pattern = {
            "troop_usage": Counter(),
            "lane_preference": Counter(),
            "win_rate": 0,
            "games_analyzed": len(battles)
        }
        
        wins = 0
        for battle in battles:
            # Extract winner and strategy position (team1 or team2)
            winner = battle.get("winner")
            team1_file = battle.get("team1_file", "")
            team2_file = battle.get("team2_file", "")
            
            team1_id = team1_file
            if team1_file.endswith('.py'):
                team1_id = team1_file[:-3]
                
            team2_id = team2_file
            if team2_file.endswith('.py'):
                team2_id = team2_file[:-3]
            
            # Count wins
            if (team1_id == strategy_id and winner == battle.get("team1_name")) or \
               (team2_id == strategy_id and winner == battle.get("team2_name")):
                wins += 1
        
        # Calculate win rate
        if battles:
            pattern["win_rate"] = wins / len(battles)
        
        return pattern
    
    def generate_counter_strategy(self, opponent_id, base_strategy=None):
        """
        Generate a counter strategy for a specific opponent.
        
        Args:
            opponent_id: ID of the opponent strategy
            base_strategy: Optional base strategy to adapt
            
        Returns:
            Counter strategy dictionary
        """
        # Start with a base template
        if base_strategy:
            counter = copy.deepcopy(base_strategy)
        else:
            # Choose a random base strategy
            template_key = random.choice(list(self.base_strategies.keys()))
            counter = copy.deepcopy(self.base_strategies[template_key])
        
        # Add basic counter strategy metadata
        counter["id"] = f"counter_{str(uuid.uuid4())[:8]}"
        counter["name"] = f"counter_{opponent_id}"
        counter["counter_target"] = opponent_id
        
        # Analyze opponent battles if we have metrics
        opponent_analysis = self.analyze_strategy(opponent_id) if self.battle_metrics else None
        
        # Default counter parameters
        counter_params = {
            "lane_preference": "adaptive",
            "troop_priority": ["dragon", "wizard", "valkyrie", "musketeer", 
                              "knight", "archer", "minion", "skeleton"],
            "counter_weights": {
                "air_vs_ground": random.uniform(1.0, 2.0),
                "splash_vs_group": random.uniform(1.0, 2.0),
                "tank_priority": random.uniform(1.0, 2.0)
            },
            "defensive_trigger": random.uniform(0.4, 0.7),
            "counter_rationale": "Adaptive counter strategy"
        }
        
        # Apply counter parameters
        counter.update(counter_params)
        
        # Add counter strategy metadata
        counter["counter_strategy"] = {
            "is_counter": True,
            "target": opponent_id,
            "rationale": counter_params.get("counter_rationale", "Adaptive counter-strategy"),
            "adaptive": True
        }
        
        # Initialize metrics
        counter["metrics"] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "wilson_score": 0
        }
        
        return counter
    
    def create_adaptive_strategy(self, base_strategy=None):
        """
        Create a strategy that can adapt to different opponents during play.
        
        Args:
            base_strategy: Optional base strategy to adapt
            
        Returns:
            Adaptive strategy dictionary
        """
        # Start with a base strategy if provided, otherwise use a random one
        if base_strategy:
            strategy = copy.deepcopy(base_strategy)
        else:
            # Pick a random base strategy
            base_key = random.choice(list(self.base_strategies.keys()))
            strategy = copy.deepcopy(self.base_strategies[base_key])
        
        # Add adaptive components
        strategy["id"] = f"adaptive_{str(uuid.uuid4())[:8]}"
        strategy["name"] = "adaptive_strategy"
        
        # Make sure we have troop priority
        if "troop_priority" not in strategy:
            strategy["troop_priority"] = ["dragon", "wizard", "valkyrie", "musketeer", 
                                        "knight", "archer", "minion", "skeleton"]
        
        # Make sure we have lane preference
        if "lane_preference" not in strategy:
            strategy["lane_preference"] = "adaptive"
        
        # Adaptive parameters
        strategy["adaptive_strategy"] = {
            "enabled": True,
            "analyze_opponent": True,
            "adaptation_threshold": random.uniform(0.3, 0.7),
            "lane_adaptation": True,
            "troop_adaptation": True,
            "initial_observation_time": random.randint(5, 15),
            "adaptation_rationale": "Dynamic strategy that adapts to opponent patterns"
        }
        
        # Add parameters for counter weights if missing
        if "counter_weights" not in strategy:
            strategy["counter_weights"] = {
                "air_vs_ground": random.uniform(0.8, 1.5),
                "splash_vs_group": random.uniform(0.8, 1.5),
                "tank_priority": random.uniform(0.8, 1.5)
            }
        
        # Add defensive trigger if missing
        if "defensive_trigger" not in strategy:
            strategy["defensive_trigger"] = random.uniform(0.4, 0.8)
        
        # Initialize metrics
        strategy["metrics"] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "wilson_score": 0
        }
        
        return strategy
    
    def generate_adaptive_strategies(self, count=2, base_strategies=None):
        """
        Generate multiple adaptive strategies.
        
        Args:
            count: Number of strategies to generate
            base_strategies: Optional list of base strategies to adapt from
            
        Returns:
            List of adaptive strategies
        """
        strategies = []
        for _ in range(count):
            # Choose a random base strategy if available
            base = None
            if base_strategies:
                if isinstance(base_strategies, dict):
                    bases = list(base_strategies.values())
                    if bases:
                        base = random.choice(bases)
                elif isinstance(base_strategies, list) and base_strategies:
                    base = random.choice(base_strategies)
            
            # Create adaptive strategy
            strategy = self.create_adaptive_strategy(base)
            strategies.append(strategy)
        
        return strategies
    
    def generate_counter_strategies(self, opponent_strategies, count=2, base_strategies=None):
        """
        Generate multiple counter strategies.
        
        Args:
            opponent_strategies: List of strategies to counter
            count: Number of strategies to generate
            base_strategies: Optional list of base strategies to adapt from
            
        Returns:
            List of counter strategies
        """
        strategies = []
        
        # Convert to list if it's a dictionary
        if isinstance(opponent_strategies, dict):
            opponents = list(opponent_strategies.values())
        else:
            opponents = opponent_strategies
        
        # Make sure we have opponents
        if not opponents:
            return strategies
        
        # Generate counter strategies
        for i in range(min(count, len(opponents))):
            opponent = opponents[i]
            opponent_id = opponent.get("id", f"unknown_{i}")
            
            # Choose a random base strategy if available
            base = None
            if base_strategies:
                if isinstance(base_strategies, dict):
                    bases = list(base_strategies.values())
                    if bases:
                        base = random.choice(bases)
                elif isinstance(base_strategies, list) and base_strategies:
                    base = random.choice(base_strategies)
            
            # Create counter strategy
            strategy = self.generate_counter_strategy(opponent_id, base)
            strategies.append(strategy)
        
        return strategies

# Function to update strategy_generator.py code generation for adaptive strategies
def get_adaptive_strategy_code_segment():
    """Return code segment for adaptive strategy implementation in generated code."""
    return """
    def pre_deploy_logic(self, data):
        '''Analysis and adaptation logic that runs before deployment.'''
        # Initialize tower references if first run
        if not hasattr(self, 'my_tower') or not self.my_tower:
            self.my_tower = data.get('my_tower')
            self.opponent_tower = data.get('opponent_tower')
        
        # Track game time (incremented each frame)
        if not hasattr(self, 'game_time'):
            self.game_time = 0
        self.game_time += 1
        
        # Initialize opponent analysis tracking
        if not hasattr(self, 'opponent_troop_counts'):
            self.opponent_troop_counts = {}
            self.opponent_lane_counts = {}
            self.opponent_deploy_times = []
            self.opponent_air_ground_ratio = 0.5
            self.opponent_splash_single_ratio = 0.5
            self.adapted_troop_priority = None
            self.adapted_lane_preference = None
            
            # Troop type classifications
            self.troop_types = {
                "air": ["dragon", "minion"],
                "ground": ["knight", "archer", "valkyrie", "wizard", "musketeer", "prince", "giant", "skeleton", "bomber"],
                "splash": ["wizard", "valkyrie", "bomber"],
                "single": ["knight", "archer", "musketeer", "prince", "giant", "skeleton", "dragon", "minion"],
                "tank": ["knight", "giant", "prince", "valkyrie"],
                "dps": ["archer", "wizard", "musketeer", "dragon", "minion", "skeleton", "bomber"]
            }
            
            # Troop counters
            self.troop_counters = {
                "dragon": ["archer", "musketeer", "wizard"],
                "minion": ["archer", "musketeer", "wizard"],
                "wizard": ["knight", "giant", "prince"],
                "musketeer": ["knight", "valkyrie", "prince"],
                "valkyrie": ["minion", "dragon", "skeleton"],
                "knight": ["skeleton", "minion", "dragon"],
                "archer": ["knight", "giant", "valkyrie"],
                "prince": ["skeleton", "archer", "wizard"],
                "giant": ["skeleton", "minion", "valkyrie"],
                "skeleton": ["wizard", "valkyrie", "bomber"]
            }
        
        # Track opponent troops for analysis
        opponent_troops = data.get('opponent_troops', [])
        for troop in opponent_troops:
            # Skip if we've already counted this troop
            if not hasattr(troop, 'id') or troop.id in self.opponent_troop_counts:
                continue
                
            # Add to our analysis
            troop_type = troop.name if hasattr(troop, 'name') else "unknown"
            self.opponent_troop_counts[troop.id] = troop_type
            
            # Track lane based on x position
            if hasattr(troop, 'position') and troop.position:
                x = troop.position[0]
                if x < -20:
                    lane = "left"
                elif x > 20:
                    lane = "right"
                else:
                    lane = "center"
                    
                # Count lane usage
                self.opponent_lane_counts[lane] = self.opponent_lane_counts.get(lane, 0) + 1
        
        # Only adapt after observation period
        if self.game_time < 10:  # Wait at least 10 frames
            return
            
        # Perform adaptation if enabled
        if hasattr(self, 'adaptive_strategy') and self.adaptive_strategy.get('enabled', False):
            self._adapt_to_opponent()
    
    def _adapt_to_opponent(self):
        '''Adapt strategy based on opponent analysis.'''
        # Skip if we don't have enough data
        if not self.opponent_troop_counts:
            return
            
        # Calculate opponent air/ground ratio
        air_count = 0
        ground_count = 0
        splash_count = 0
        single_count = 0
        
        for troop_type in self.opponent_troop_counts.values():
            if troop_type in self.troop_types["air"]:
                air_count += 1
            elif troop_type in self.troop_types["ground"]:
                ground_count += 1
                
            if troop_type in self.troop_types["splash"]:
                splash_count += 1
            elif troop_type in self.troop_types["single"]:
                single_count += 1
        
        total_troops = len(self.opponent_troop_counts)
        if total_troops > 0:
            self.opponent_air_ground_ratio = air_count / total_troops
            self.opponent_splash_single_ratio = splash_count / total_troops if splash_count > 0 else 0
        
        # Adapt troop priority based on opponent troops
        if hasattr(self, 'adaptive_strategy') and self.adaptive_strategy.get('troop_adaptation', True):
            self._adapt_troop_priority()
        
        # Adapt lane preference based on opponent lanes
        if hasattr(self, 'adaptive_strategy') and self.adaptive_strategy.get('lane_adaptation', True) and self.opponent_lane_counts:
            self._adapt_lane_preference()
    
    def _adapt_troop_priority(self):
        '''Adapt troop priority based on opponent analysis.'''
        # Start with the base priority
        new_priority = list(self.troop_priority)
        
        # If opponent uses mostly air troops, prioritize anti-air
        if self.opponent_air_ground_ratio > 0.6:
            # Move anti-air troops to the front
            anti_air = ["archer", "musketeer", "wizard"]
            for troop in reversed(anti_air):
                if troop in new_priority:
                    new_priority.remove(troop)
                    new_priority.insert(0, troop)
        
        # If opponent uses mostly splash damage, prioritize spread/air units
        if self.opponent_splash_single_ratio > 0.5:
            # Move air and tank troops up
            air_tanks = ["dragon", "minion", "knight", "giant", "prince"]
            for troop in reversed(air_tanks):
                if troop in new_priority:
                    new_priority.remove(troop)
                    new_priority.insert(0, troop)
        
        # Only adapt if the new priority is different
        if new_priority != self.troop_priority:
            self.adapted_troop_priority = new_priority
    
    def _adapt_lane_preference(self):
        '''Adapt lane preference based on opponent deployments.'''
        # Find most common opponent lane
        most_common_lane = max(self.opponent_lane_counts.items(), key=lambda x: x[1])[0]
        
        # Counter the opponent's lane preference
        if most_common_lane == "left":
            new_preference = "left"  # Attack same lane to intercept
        elif most_common_lane == "right":
            new_preference = "right"  # Attack same lane to intercept
        elif most_common_lane == "center":
            new_preference = "split"  # Split to surround center approach
        else:
            new_preference = "adaptive"
        
        # Only adapt if different from current
        if new_preference != self.lane_preference:
            self.adapted_lane_preference = new_preference
"""