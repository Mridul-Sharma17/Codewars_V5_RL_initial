"""
Adaptive strategy templates for tower defense reinforcement learning system.
Provides templates for different strategy types that can adapt to opponent behavior.
"""

import random
from typing import Dict, Any, List, Optional

class AdaptiveTemplates:
    """
    Collection of adaptive strategy templates that can be used to generate
    more intelligent and responsive strategy code.
    """
    
    @staticmethod
    def generate_template_code(strategy_type: str, params: Dict[str, Any]) -> str:
        """
        Generate code for a specific template type with the given parameters.
        
        Args:
            strategy_type: Type of strategy template to generate
            params: Parameters to use in the template
            
        Returns:
            Python code string for the strategy
        """
        template_methods = {
            "counter_picker": AdaptiveTemplates.counter_picker_template,
            "lane_adaptor": AdaptiveTemplates.lane_adaptor_template,
            "elixir_optimizer": AdaptiveTemplates.elixir_optimizer_template,
            "phase_shifter": AdaptiveTemplates.phase_shifter_template,
            "pattern_recognition": AdaptiveTemplates.pattern_recognition_template,
            "troop_synergy": AdaptiveTemplates.troop_synergy_template,
            "adaptive": AdaptiveTemplates.fully_adaptive_template,
            "baseline": AdaptiveTemplates.baseline_template
        }
        
        if strategy_type in template_methods:
            return template_methods[strategy_type](params)
        else:
            # Default to baseline if template type not found
            print(f"Warning: Strategy type '{strategy_type}' not found. Using baseline template.")
            return AdaptiveTemplates.baseline_template(params)
    
    @staticmethod
    def counter_picker_template(params: Dict[str, Any]) -> str:
        """
        Generate a counter-picker strategy template that analyzes opponent troop types
        and deploys appropriate counters.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        # Extract specific parameters
        counter_weights = params.get("counter_weights", {})
        air_vs_ground = counter_weights.get("air_vs_ground", 1.2)
        splash_vs_group = counter_weights.get("splash_vs_group", 1.5)
        
        code = f"""from teams.helper_function import Troops, Utils
import random

team_name = "{params.get('name', 'Counter_Picker')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{params.get('lane_preference', 'adaptive')[0]},e:{params.get('elixir_thresholds', {}).get('min_deploy', 4):.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:strategy"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Initialize opponent analysis storage if it doesn't exist
    if not hasattr(logic, 'opponent_analysis'):
        logic.opponent_analysis = {{
            'air_count': 0,
            'ground_count': 0,
            'group_count': 0,    # Multiple troops close together
            'solo_count': 0,     # Single high-health units
            'seen_troops': set()
        }}
    
    # Analyze opponent troops to understand their strategy
    analyze_opponent_troops(opp_troops)
    
    # Determine if we should deploy troops
    if should_deploy(my_tower, opp_tower, opp_troops, params):
        # Choose best troop and position based on counter-picking strategy
        troop, position = choose_counter_troop(my_tower, opp_troops, params)
        if troop:
            deploy_troop(troop, position)

def analyze_opponent_troops(opp_troops):
    \"\"\"Analyze opponent troops to determine their strategy\"\"\"
    # Reset counts for this frame
    air_count = 0
    ground_count = 0
    
    # Count current troops by type
    for troop in opp_troops:
        # Add to seen troops set
        logic.opponent_analysis['seen_troops'].add(troop.name)
        
        # Count by type
        if troop.type == "air":
            air_count += 1
        elif troop.type == "ground":
            ground_count += 1
    
    # Update running counts with decay (0.9 weight to prior counts)
    logic.opponent_analysis['air_count'] = 0.9 * logic.opponent_analysis['air_count'] + 0.1 * air_count
    logic.opponent_analysis['ground_count'] = 0.9 * logic.opponent_analysis['ground_count'] + 0.1 * ground_count
    
    # Detect if opponent uses grouped deployments (check if troops are close together)
    position_groups = []
    for troop in opp_troops:
        added = False
        for group in position_groups:
            for group_troop in group:
                distance = Utils.calculate_distance(troop, group_troop, type_troop=True)
                if distance < 10:  # Close enough to be considered grouped
                    group.append(troop)
                    added = True
                    break
            if added:
                break
        
        if not added:
            position_groups.append([troop])
    
    # Count groups and solo units
    group_count = sum(1 for group in position_groups if len(group) > 1)
    solo_count = sum(1 for group in position_groups if len(group) == 1)
    
    # Update group analysis with decay
    logic.opponent_analysis['group_count'] = 0.9 * logic.opponent_analysis['group_count'] + 0.1 * group_count
    logic.opponent_analysis['solo_count'] = 0.9 * logic.opponent_analysis['solo_count'] + 0.1 * solo_count

def parse_signal(signal):
    \"\"\"Parse strategy parameters from team_signal\"\"\"
    # Default parameters
    params = {{
        "lane": "{params.get('lane_preference', 'adaptive')}",
        "min_elixir": {params.get('elixir_thresholds', {}).get('min_deploy', 4)},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)}
    }}
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    \"\"\"Track opponent troop types in team_signal\"\"\"
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def should_deploy(my_tower, opp_tower, opp_troops, params):
    \"\"\"Decide whether to deploy troops based on strategy and game state\"\"\"
    # Check if we have enough elixir
    if my_tower.total_elixir < params["min_elixir"]:
        return False
    
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
    
    # Always deploy early in the game
    if my_tower.game_timer < 300:
        return True
    
    # Deploy based on elixir availability
    if my_tower.total_elixir >= 8:  # Lots of elixir, deploy something
        return True
    elif my_tower.total_elixir >= params["min_elixir"] + 2:  # Extra elixir available
        return random.random() > 0.3  # 70% chance to deploy
    
    # Default: deploy if we meet minimum elixir
    return my_tower.total_elixir >= params["min_elixir"]

def choose_counter_troop(my_tower, opp_troops, params):
    \"\"\"Choose the best troop to counter the opponent's strategy\"\"\"
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Base troop scores
    troop_scores = {{
        "Wizard": 6 if "Wizard" in available_troops else 0,
        "Dragon": 7 if "Dragon" in available_troops else 0,
        "Musketeer": 5 if "Musketeer" in available_troops else 0,
        "Valkyrie": 6 if "Valkyrie" in available_troops else 0,
        "Knight": 4 if "Knight" in available_troops else 0,
        "Archer": 3 if "Archer" in available_troops else 0,
        "Barbarian": 2 if "Barbarian" in available_troops else 0,
        "Minion": 3 if "Minion" in available_troops else 0
    }}
    
    # Apply counter-picking logic based on opponent analysis
    if logic.opponent_analysis['air_count'] > logic.opponent_analysis['ground_count'] * {air_vs_ground}:
        # Heavy air presence - prioritize anti-air troops
        troop_scores["Musketeer"] += 5
        troop_scores["Wizard"] += 4
        troop_scores["Archer"] += 3
    elif logic.opponent_analysis['ground_count'] > logic.opponent_analysis['air_count'] * {air_vs_ground}:
        # Heavy ground presence - prioritize air troops and splash damage
        troop_scores["Dragon"] += 5
        troop_scores["Wizard"] += 4
        troop_scores["Valkyrie"] += 3
        troop_scores["Minion"] += 3
    
    # Counter group deployments with splash damage
    if logic.opponent_analysis['group_count'] > logic.opponent_analysis['solo_count'] * {splash_vs_group}:
        troop_scores["Wizard"] += 4
        troop_scores["Valkyrie"] += 4
        troop_scores["Dragon"] += 3
    
    # If in defensive mode, prioritize different troops
    if defensive_mode:
        troop_scores["Valkyrie"] += 2
        troop_scores["Knight"] += 2
        troop_scores["Wizard"] += 2
    
    # If opponent has specific troops we've seen, counter them
    if "Dragon" in logic.opponent_analysis['seen_troops']:
        troop_scores["Musketeer"] += 2
        troop_scores["Archer"] += 1
    
    if "Knight" in logic.opponent_analysis['seen_troops'] or "Valkyrie" in logic.opponent_analysis['seen_troops']:
        troop_scores["Minion"] += 2
        troop_scores["Dragon"] += 1
    
    # Choose the highest scoring available troop
    best_score = -1
    chosen_troop = None
    
    for troop, score in troop_scores.items():
        if score > best_score and troop in available_troops:
            best_score = score
            chosen_troop = troop
    
    # Fallback to first available troop if none selected
    if not chosen_troop and available_troops:
        chosen_troop = available_troops[0]
    
    # Determine position based on deployment strategy
    x_pos, y_pos = determine_position(params, defensive_mode, opp_troops)
    
    return chosen_troop, (x_pos, y_pos)

def determine_position(params, defensive_mode, opp_troops):
    \"\"\"Determine the best position for troop deployment\"\"\"
    # Default parameters
    lane = params.get("lane")
    
    # Dynamic lane selection if adaptive
    if lane == "adaptive" and hasattr(logic, 'opponent_analysis'):
        # Analyze where opponent troops are concentrated
        left_count = 0
        right_count = 0
        center_count = 0
        
        for troop in opp_troops:
            x_pos = troop.position[0]
            if x_pos < -10:
                left_count += 1
            elif x_pos > 10:
                right_count += 1
            else:
                center_count += 1
        
        # Counter by deploying in the opposite lane when attacking
        # or same lane when defending
        if defensive_mode:
            # When defending, deploy to the same lane as opponent
            if left_count > right_count and left_count > center_count:
                lane = "left"
            elif right_count > left_count and right_count > center_count:
                lane = "right"
            else:
                lane = "center"
        else:
            # When attacking, deploy to the opposite lane
            if left_count > right_count and left_count > center_count:
                lane = "right"  # Opposite of left
            elif right_count > left_count and right_count > center_count:
                lane = "left"   # Opposite of right
            else:
                # If center or balanced, choose randomly between left and right
                lane = random.choice(["left", "right"])
    
    # Set x position based on lane preference
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right for split pushing
        if not hasattr(determine_position, 'last_lane'):
            determine_position.last_lane = "left"
        
        if determine_position.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            determine_position.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            determine_position.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on defensive mode
    y_pos = params["y_defensive"] if defensive_mode else params["y_default"]
    
    return x_pos, y_pos

def deploy_troop(troop, position):
    \"\"\"Deploy the selected troop at the given position\"\"\"
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code
    
    @staticmethod
    def lane_adaptor_template(params: Dict[str, Any]) -> str:
        """
        Generate a lane-adaptor strategy template that analyzes and responds
        to opponent lane preferences.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        # Extract specific parameters
        lane_pref = params.get('lane_preference', 'adaptive')
        
        code = f"""from teams.helper_function import Troops, Utils
import random

team_name = "{params.get('name', 'Lane_Adaptor')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{lane_pref[0]},e:{params.get('elixir_thresholds', {}).get('min_deploy', 4):.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:strategy"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Initialize lane analysis if it doesn't exist
    if not hasattr(logic, 'lane_counts'):
        logic.lane_counts = {{"left": 0, "center": 0, "right": 0}}
        logic.historical_lanes = []
        logic.current_preferred_lane = None
    
    # Track lane preferences
    track_lane_preferences(opp_troops)
    
    # Analyze whether to deploy troops based on elixir and game state
    if should_deploy(my_tower, opp_tower, opp_troops, params):
        # Choose best troop and position based on strategy
        troop, position = choose_troop_and_position(my_tower, opp_troops, params)
        if troop:
            deploy_troop(troop, position)

def track_lane_preferences(opp_troops):
    \"\"\"Analyze and track opponent lane preferences\"\"\"
    # Count troops in each lane for this frame
    current_counts = {{"left": 0, "center": 0, "right": 0}}
    
    for troop in opp_troops:
        x_pos = troop.position[0]
        if x_pos < -10:
            current_counts["left"] += 1
        elif x_pos > 10:
            current_counts["right"] += 1
        else:
            current_counts["center"] += 1
    
    # Update running counts with decay (0.9 weight to prior counts)
    logic.lane_counts["left"] = 0.9 * logic.lane_counts["left"] + 0.1 * current_counts["left"]
    logic.lane_counts["center"] = 0.9 * logic.lane_counts["center"] + 0.1 * current_counts["center"]
    logic.lane_counts["right"] = 0.9 * logic.lane_counts["right"] + 0.1 * current_counts["right"]
    
    # Determine the opponent's currently preferred lane
    max_lane = max(logic.lane_counts.items(), key=lambda x: x[1])
    
    # Only consider as preferred if there's a clear preference
    total = sum(logic.lane_counts.values())
    if total > 0 and max_lane[1] / total > 0.4:  # At least 40% preference
        preferred_lane = max_lane[0]
        
        # Update the history
        logic.historical_lanes.append(preferred_lane)
        if len(logic.historical_lanes) > 5:  # Keep only last 5
            logic.historical_lanes.pop(0)
            
        # Update current preferred lane based on recent history
        if logic.historical_lanes.count("left") > len(logic.historical_lanes) // 2:
            logic.current_preferred_lane = "left"
        elif logic.historical_lanes.count("right") > len(logic.historical_lanes) // 2:
            logic.current_preferred_lane = "right"
        elif logic.historical_lanes.count("center") > len(logic.historical_lanes) // 2:
            logic.current_preferred_lane = "center"
        else:
            # No clear preference in history
            logic.current_preferred_lane = preferred_lane

def parse_signal(signal):
    \"\"\"Parse strategy parameters from team_signal\"\"\"
    # Default parameters
    params = {{
        "lane": "{lane_pref}",
        "min_elixir": {params.get('elixir_thresholds', {}).get('min_deploy', 4)},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)}
    }}
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    \"\"\"Track opponent troop types in team_signal\"\"\"
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def should_deploy(my_tower, opp_tower, opp_troops, params):
    \"\"\"Decide whether to deploy troops based on strategy and game state\"\"\"
    # Check if we have enough elixir
    if my_tower.total_elixir < params["min_elixir"]:
        return False
    
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
    
    # Always deploy early in the game
    if my_tower.game_timer < 300:
        return True
    
    # Deploy based on elixir availability
    if my_tower.total_elixir >= 8:  # Lots of elixir, deploy something
        return True
    elif my_tower.total_elixir >= params["min_elixir"] + 2:  # Extra elixir available
        return random.random() > 0.3  # 70% chance to deploy
    
    # Default: deploy if we meet minimum elixir
    return my_tower.total_elixir >= params["min_elixir"]

def choose_troop_and_position(my_tower, opp_troops, params):
    \"\"\"Choose the best troop to deploy and the position\"\"\"
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Choose troop based on strategy
    chosen_troop = None
    
    # Simple heuristic: choose the first available highest-damage troop
    troop_scores = {{
        "Wizard": 10 if "Wizard" in available_troops else 0,
        "Dragon": 9 if "Dragon" in available_troops else 0,
        "Musketeer": 8 if "Musketeer" in available_troops else 0,
        "Valkyrie": 7 if "Valkyrie" in available_troops else 0,
        "Knight": 6 if "Knight" in available_troops else 0,
        "Archer": 5 if "Archer" in available_troops else 0,
        "Barbarian": 4 if "Barbarian" in available_troops else 0,
        "Minion": 3 if "Minion" in available_troops else 0
    }}
    
    # Adjust scores based on opponent troops
    if opp_troops:
        air_count = sum(1 for troop in opp_troops if troop.type == "air")
        ground_count = sum(1 for troop in opp_troops if troop.type == "ground")
        
        if air_count > ground_count:
            # Prioritize anti-air troops
            troop_scores["Wizard"] += 3
            troop_scores["Musketeer"] += 3
            troop_scores["Archer"] += 2
        else:
            # Prioritize anti-ground troops
            troop_scores["Valkyrie"] += 3
            troop_scores["Knight"] += 2
            troop_scores["Barbarian"] += 2
    
    # Choose the highest scoring available troop
    best_score = -1
    for troop, score in troop_scores.items():
        if score > best_score and troop in available_troops:
            best_score = score
            chosen_troop = troop
    
    # Fallback to first available troop if none selected
    if not chosen_troop and available_troops:
        chosen_troop = available_troops[0]
    
    # Determine position based on adaptive lane strategy
    lane = params["lane"]
    
    # If we're using the adaptive lane strategy, determine lane based on opponent
    if lane == "adaptive" and hasattr(logic, 'current_preferred_lane'):
        if defensive_mode:
            # When defensive, deploy in the same lane as the opponent
            lane = logic.current_preferred_lane
        else:
            # When offensive, deploy in the opposite lane
            if logic.current_preferred_lane == "left":
                lane = "right"
            elif logic.current_preferred_lane == "right":
                lane = "left"
            else:
                # If opponent is center, choose randomly
                lane = random.choice(["left", "right"])
    
    # Set deployment position based on lane
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right for split pushing
        if not hasattr(choose_troop_and_position, 'last_lane'):
            choose_troop_and_position.last_lane = "left"
        
        if choose_troop_and_position.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            choose_troop_and_position.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            choose_troop_and_position.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on defensive mode
    y_pos = params["y_defensive"] if defensive_mode else params["y_default"]
    
    return chosen_troop, (x_pos, y_pos)

def deploy_troop(troop, position):
    '''Deploy the selected troop at the given position'''
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code
    
    @staticmethod
    def elixir_optimizer_template(params: Dict[str, Any]) -> str:
        """
        Generate an elixir-optimizer strategy template that maximizes elixir usage efficiency.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        # Extract specific parameters
        elixir_thresholds = params.get("elixir_thresholds", {})
        min_deploy = elixir_thresholds.get("min_deploy", 4)
        save_threshold = elixir_thresholds.get("save_threshold", 7)
        emergency_threshold = elixir_thresholds.get("emergency_threshold", 2)
        
        # Get timing parameters
        timing_params = params.get("timing_parameters", {})
        deployment_interval = timing_params.get("deployment_interval", 1.5)
        burst_threshold = timing_params.get("burst_threshold", 8)
        patience_factor = timing_params.get("patience_factor", 0.7)
        
        code = f"""from teams.helper_function import Troops, Utils
import random

team_name = "{params.get('name', 'Elixir_Optimizer')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{params.get('lane_preference', 'center')[0]},e:{min_deploy:.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:optimize"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Initialize elixir management trackers if they don't exist
    if not hasattr(logic, 'elixir_history'):
        logic.elixir_history = []
        logic.last_deployment_time = 0
        logic.opponent_deploy_times = []
        logic.last_elixir = 0
        logic.elixir_efficiency = 0
        logic.optimal_elixir = {save_threshold}  # Initial optimal value
    
    # Track elixir spending efficiency
    track_elixir_efficiency(my_tower, opp_troops)
    
    # Determine if we should deploy troops based on elixir optimization
    if should_deploy_optimal(my_tower, opp_tower, opp_troops, params):
        # Choose best troop and position
        troop, position = choose_optimal_troop(my_tower, opp_troops, params)
        if troop:
            deploy_troop(troop, position)
            # Record when we deployed
            logic.last_deployment_time = my_tower.game_timer

def track_elixir_efficiency(my_tower, opp_troops):
    \"\"\"Track and optimize elixir usage efficiency\"\"\"
    # Add current elixir level to history
    logic.elixir_history.append(my_tower.total_elixir)
    if len(logic.elixir_history) > 100:  # Keep history manageable
        logic.elixir_history.pop(0)
    
    # Calculate elixir generation rate
    if logic.last_elixir > 0:
        elixir_gain = my_tower.total_elixir - logic.last_elixir
        # If elixir decreased, it was likely spent on deployment
        if elixir_gain < 0:
            # Don't count as negative efficiency
            pass
        else:
            # Track our elixir efficiency (gain per frame)
            # Higher values mean we're generating elixir faster than spending it
            logic.elixir_efficiency = 0.9 * logic.elixir_efficiency + 0.1 * elixir_gain
    
    # Record current elixir for next frame
    logic.last_elixir = my_tower.total_elixir
    
    # Detect when opponent deploys troops (new troops appear)
    if len(opp_troops) > 0 and not hasattr(logic, 'last_opponent_count'):
        logic.last_opponent_count = 0
    
    if hasattr(logic, 'last_opponent_count'):
        if len(opp_troops) > logic.last_opponent_count:
            # Enemy deployed troops
            logic.opponent_deploy_times.append(my_tower.game_timer)
            if len(logic.opponent_deploy_times) > 10:
                logic.opponent_deploy_times.pop(0)
        
        logic.last_opponent_count = len(opp_troops)
    
    # Adjust optimal elixir level based on game state
    adjust_optimal_elixir(my_tower)

def adjust_optimal_elixir(my_tower):
    \"\"\"Dynamically adjust the optimal elixir level based on game state\"\"\"
    # Default value
    optimal = {save_threshold}
    
    # Calculate average opponent deploy interval if we have data
    if len(logic.opponent_deploy_times) >= 2:
        intervals = [logic.opponent_deploy_times[i] - logic.opponent_deploy_times[i-1] 
                    for i in range(1, len(logic.opponent_deploy_times))]
        avg_interval = sum(intervals) / len(intervals)
        
        # If opponent deploys frequently, maintain higher elixir reserve
        if avg_interval < 100:  # Quick deployments
            optimal = max({save_threshold}, {burst_threshold} - 1)
        elif avg_interval > 300:  # Slow deployments
            optimal = max({min_deploy} + 1, {save_threshold} - 1)
    
    # Game phase adjustments
    if my_tower.game_timer < 300:  # Early game
        optimal = min(optimal, {save_threshold} - 1)  # Be slightly more aggressive
    elif my_tower.game_timer > 1200:  # Late game
        optimal = max(optimal, {save_threshold})  # Be slightly more conservative
    
    # Update our optimal elixir target
    logic.optimal_elixir = optimal

def parse_signal(signal):
    \"\"\"Parse strategy parameters from team_signal\"\"\"
    # Default parameters
    params = {{
        "lane": "{params.get('lane_preference', 'center')}",
        "min_elixir": {min_deploy},
        "save_threshold": {save_threshold},
        "emergency_threshold": {emergency_threshold},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)},
        "deployment_interval": {deployment_interval},
        "patience_factor": {patience_factor}
    }}
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    \"\"\"Track opponent troop types in team_signal\"\"\"
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def should_deploy_optimal(my_tower, opp_tower, opp_troops, params):
    \"\"\"Decide whether to deploy troops based on elixir optimization\"\"\"
    # Emergency deployment when opponent troops are approaching
    if opp_troops and len(opp_troops) >= 2:
        # Emergency mode - deploy if above emergency threshold
        return my_tower.total_elixir >= params["emergency_threshold"]
    
    # Check how long since last deployment
    time_since_deploy = my_tower.game_timer - logic.last_deployment_time
    min_interval = params["deployment_interval"] * 60  # Convert to frames
    
    # If we haven't waited long enough, don't deploy yet
    if time_since_deploy < min_interval:
        # Unless we have excess elixir
        if my_tower.total_elixir < {burst_threshold}:
            return False
    
    # If we have reached our optimal elixir level, deploy
    if my_tower.total_elixir >= logic.optimal_elixir:
        # Add randomness based on patience factor
        deploy_chance = (my_tower.total_elixir - logic.optimal_elixir) * params["patience_factor"] + 0.3
        return random.random() < min(1.0, deploy_chance)
    
    # If we're in burst mode (lots of elixir), always deploy
    if my_tower.total_elixir >= {burst_threshold}:
        return True
    
    # Default: don't deploy until we reach optimal elixir
    return False

def choose_optimal_troop(my_tower, opp_troops, params):
    \"\"\"Choose the most elixir-efficient troop for the situation\"\"\"
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Troop costs and base scores
    troop_data = {{
        "Wizard": {{"cost": 5, "score": 8}},
        "Dragon": {{"cost": 4, "score": 7}},
        "Musketeer": {{"cost": 4, "score": 6}},
        "Valkyrie": {{"cost": 4, "score": 7}},
        "Knight": {{"cost": 3, "score": 5}},
        "Archer": {{"cost": 3, "score": 4}},
        "Barbarian": {{"cost": 3, "score": 3}},
        "Minion": {{"cost": 3, "score": 4}}
    }}
    
    # Calculate efficiency scores (value per elixir)
    troop_scores = {{}}
    for troop, data in troop_data.items():
        if troop in available_troops:
            # Base efficiency is score / cost
            efficiency = data["score"] / data["cost"]
            
            # Adjust for our current elixir situation
            if my_tower.total_elixir >= {burst_threshold}:
                # In burst mode, prefer higher damage troops even if less efficient
                adjusted_score = data["score"] * efficiency
            elif my_tower.total_elixir <= params["emergency_threshold"] + 1:
                # In emergency mode, prefer cheaper troops
                adjusted_score = efficiency * (1 + (5 - data["cost"]) * 0.2)
            else:
                # Normal mode - balance efficiency with power
                adjusted_score = efficiency
            
            troop_scores[troop] = adjusted_score
    
    # Account for opponent troops when adjusting scores
    if opp_troops:
        air_count = sum(1 for troop in opp_troops if troop.type == "air")
        ground_count = sum(1 for troop in opp_troops if troop.type == "ground")
        
        if air_count > ground_count:
            # Prioritize anti-air troops
            for troop in ["Wizard", "Musketeer", "Archer"]:
                if troop in troop_scores:
                    troop_scores[troop] *= 1.5
        else:
            # Prioritize anti-ground troops
            for troop in ["Valkyrie", "Knight", "Barbarian"]:
                if troop in troop_scores:
                    troop_scores[troop] *= 1.5
    
    # Choose the highest scoring available troop
    if troop_scores:
        chosen_troop = max(troop_scores.items(), key=lambda x: x[1])[0]
    else:
        # Fallback to first available troop
        chosen_troop = available_troops[0]
    
    # Determine position based on lane preference
    lane = params["lane"]
    
    # Set deployment position based on lane
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right for split pushing
        if not hasattr(choose_optimal_troop, 'last_lane'):
            choose_optimal_troop.last_lane = "left"
        
        if choose_optimal_troop.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            choose_optimal_troop.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            choose_optimal_troop.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on defensive mode
    y_pos = params["y_defensive"] if defensive_mode else params["y_default"]
    
    return chosen_troop, (x_pos, y_pos)

def deploy_troop(troop, position):
    \"\"\"Deploy the selected troop at the given position\"\"\"
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code
    
    @staticmethod
    def phase_shifter_template(params: Dict[str, Any]) -> str:
        """
        Generate a phase-shifter strategy template that changes tactics
        based on the game phase (early, mid, late game).
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        # Extract phase parameters
        phase_params = params.get("phase_parameters", {})
        early_game_threshold = phase_params.get("early_game_threshold", 500)
        late_game_threshold = phase_params.get("late_game_threshold", 1200)
        early_aggression = phase_params.get("early_game_aggression", 0.3)
        mid_aggression = phase_params.get("mid_game_aggression", 0.7)
        late_aggression = phase_params.get("late_game_aggression", 0.9)
        
        code = f"""from teams.helper_function import Troops, Utils
import random

team_name = "{params.get('name', 'Phase_Shifter')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{params.get('lane_preference', 'center')[0]},e:{params.get('elixir_thresholds', {}).get('min_deploy', 4):.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:phased"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Initialize phase tracking
    if not hasattr(logic, 'current_phase'):
        logic.current_phase = "early"
        logic.phase_counters = {{
            "elixir_generated": 0,
            "troops_deployed": 0,
            "damage_dealt": 0
        }}
    
    # Determine current game phase
    current_phase = determine_game_phase(my_tower, opp_tower)
    
    # Analyze whether to deploy troops based on phase-specific strategy
    if should_deploy_by_phase(my_tower, opp_tower, opp_troops, params, current_phase):
        # Choose best troop and position based on current phase
        troop, position = choose_troop_by_phase(my_tower, opp_troops, params, current_phase)
        if troop:
            deploy_troop(troop, position)
            # Track deployments
            logic.phase_counters["troops_deployed"] += 1

def determine_game_phase(my_tower, opp_tower):
    \"\"\"Determine the current game phase based on game timer and state\"\"\"
    game_timer = my_tower.game_timer
    
    # Update previous phase
    if hasattr(logic, 'current_phase'):
        previous_phase = logic.current_phase
    else:
        previous_phase = "early"
    
    # Determine phase by timer
    if game_timer < {early_game_threshold}:
        current_phase = "early"
    elif game_timer < {late_game_threshold}:
        current_phase = "mid"
    else:
        current_phase = "late"
    
    # Store current phase
    logic.current_phase = current_phase
    
    # Announce phase transition
    if current_phase != previous_phase:
        pass  # We could log phase transitions here if needed
    
    return current_phase

def parse_signal(signal):
    \"\"\"Parse strategy parameters from team_signal\"\"\"
    # Default parameters
    params = {{
        "lane": "{params.get('lane_preference', 'center')}",
        "min_elixir": {params.get('elixir_thresholds', {}).get('min_deploy', 4)},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)},
        "early_aggression": {early_aggression},
        "mid_aggression": {mid_aggression},
        "late_aggression": {late_aggression}
    }}
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    \"\"\"Track opponent troop types in team_signal\"\"\"
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def should_deploy_by_phase(my_tower, opp_tower, opp_troops, params, phase):
    \"\"\"Decide whether to deploy troops based on current game phase\"\"\"
    # Get phase-specific parameters
    aggression_level = {
        "early": params["early_aggression"],
        "mid": params["mid_aggression"],
        "late": params["late_aggression"]
    }[phase]
    
    # Phase-specific minimum elixir
    phase_min_elixir = {
        "early": params["min_elixir"] + 1,  # More conservative early
        "mid": params["min_elixir"],         # Standard in mid-game
        "late": params["min_elixir"] - 1     # More aggressive late
    }[phase]
    
    # Check if we have enough elixir for this phase
    if my_tower.total_elixir < phase_min_elixir:
        return False
    
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
    
    # Health-based deployment decisions
    my_health_percent = my_tower.health / 10000
    opponent_health_percent = opp_tower.health / 10000
    
    # If we're losing badly in late game, be more aggressive
    if phase == "late" and my_health_percent < opponent_health_percent - 0.2:
        if my_tower.total_elixir >= phase_min_elixir:
            return True
    
    # If we're winning in early game, be more conservative
    if phase == "early" and my_health_percent > opponent_health_percent + 0.2:
        if my_tower.total_elixir < phase_min_elixir + 2:
            return False
    
    # Use aggression level to determine random deployment chance
    if my_tower.total_elixir >= params["min_elixir"] + 3:  # Extra elixir available
        # Higher chance of deployment with higher aggression
        return random.random() < aggression_level
    
    # Default: deploy if we meet minimum elixir with higher chance based on aggression
    if my_tower.total_elixir >= phase_min_elixir:
        return random.random() < (aggression_level / 2)  # Lower chance with min elixir
    
    return False

def choose_troop_by_phase(my_tower, opp_troops, params, phase):
    \"\"\"Choose the best troop based on current game phase\"\"\"
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Phase-specific troop preferences
    phase_preferences = {
        "early": {
            "Wizard": 7,
            "Dragon": 5,
            "Musketeer": 8,  # Good early investment
            "Valkyrie": 6,
            "Knight": 9,     # Cost-effective early
            "Archer": 10,    # Best early option
            "Barbarian": 6,
            "Minion": 7      # Good early harassment
        },
        "mid": {
            "Wizard": 9,     # Strong in mid-game
            "Dragon": 8,
            "Musketeer": 7,
            "Valkyrie": 10,  # Best mid-game option
            "Knight": 8,
            "Archer": 6,
            "Barbarian": 5,
            "Minion": 6
        },
        "late": {
            "Wizard": 10,    # Best late-game option
            "Dragon": 9,     # Strong late game
            "Musketeer": 7,
            "Valkyrie": 8,
            "Knight": 6,
            "Archer": 5,
            "Barbarian": 7,
            "Minion": 8
        }
    }
    
    # Get preferences for current phase
    preferences = phase_preferences[phase]
    
    # Create scores for available troops
    troop_scores = {{troop: (preferences[troop] if troop in preferences else 5) 
                    for troop in available_troops}}
    
    # Adjust scores based on opponent troops
    if opp_troops:
        air_count = sum(1 for troop in opp_troops if troop.type == "air")
        ground_count = sum(1 for troop in opp_troops if troop.type == "ground")
        
        if air_count > ground_count:
            # Prioritize anti-air troops
            for troop in ["Wizard", "Musketeer", "Archer"]:
                if troop in troop_scores:
                    troop_scores[troop] += 3
        else:
            # Prioritize anti-ground troops
            for troop in ["Valkyrie", "Knight", "Barbarian"]:
                if troop in troop_scores:
                    troop_scores[troop] += 3
    
    # Adjust scores for defensive mode
    if defensive_mode:
        defensive_troops = ["Valkyrie", "Knight", "Wizard"]
        for troop in defensive_troops:
            if troop in troop_scores:
                troop_scores[troop] += 2
    
    # Choose the highest scoring available troop
    best_score = -1
    chosen_troop = None
    
    for troop, score in troop_scores.items():
        if score > best_score and troop in available_troops:
            best_score = score
            chosen_troop = troop
    
    # Fallback to first available troop if none selected
    if not chosen_troop and available_troops:
        chosen_troop = available_troops[0]
    
    # Determine position based on game phase and lane preference
    lane = params["lane"]
    
    # Phase-specific positioning
    if phase == "early":
        # In early game, deploy defensively
        y_offset = -5  # Slightly more defensive
    elif phase == "mid":
        # In mid game, use standard positioning
        y_offset = 0
    else:  # late game
        # In late game, deploy more aggressively
        y_offset = 10  # More forward
    
    # Set deployment position based on lane
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right
        if not hasattr(choose_troop_by_phase, 'last_lane'):
            choose_troop_by_phase.last_lane = "left"
        
        if choose_troop_by_phase.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            choose_troop_by_phase.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            choose_troop_by_phase.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on phase and defensive mode
    base_y = params["y_defensive"] if defensive_mode else params["y_default"]
    y_pos = base_y + y_offset
    
    # Ensure y_pos stays within reasonable bounds
    y_pos = max(10, min(y_pos, 50))
    
    return chosen_troop, (x_pos, y_pos)

def deploy_troop(troop, position):
    '''Deploy the selected troop at the given position'''
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code
    
    @staticmethod
    def pattern_recognition_template(params: Dict[str, Any]) -> str:
        """
        Generate a pattern recognition strategy template that learns and responds
        to opponent deployment patterns.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        # Extract memory parameters
        memory_params = params.get("memory_parameters", {})
        memory_length = memory_params.get("memory_length", 5)
        adaptation_rate = memory_params.get("adaptation_rate", 0.5)
        
        code = f"""from teams.helper_function import Troops, Utils
import random
import math

team_name = "{params.get('name', 'Pattern_Recognizer')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{params.get('lane_preference', 'adaptive')[0]},e:{params.get('elixir_thresholds', {}).get('min_deploy', 4):.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:pattern"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Initialize pattern memory if it doesn't exist
    if not hasattr(logic, 'pattern_memory'):
        logic.pattern_memory = {{
            'last_opponent_troops': [],  # Store last few opponent troops
            'last_opponent_positions': [],  # Store last few opponent positions
            'deployment_times': [],  # When opponent tends to deploy
            'repeat_patterns': {{}},  # Observed repeating patterns
            'our_last_deployment': None,  # Our last move
            'opponent_responses': {{}},  # How opponent responds to our moves
            'last_game_time': 0,  # Last frame's game time
            'elixir_spending_rate': 0,  # Estimated opponent elixir spending
        }}
    
    # Record the current game time
    current_time = my_tower.game_timer
    
    # Update pattern memory
    update_pattern_memory(opp_troops, current_time)
    
    # Analyze opponent patterns
    analyze_patterns()
    
    # Determine if we should deploy troops
    if should_deploy_pattern_based(my_tower, opp_tower, opp_troops, params):
        # Choose troop and position based on recognized patterns
        troop, position = choose_counter_pattern(my_tower, opp_troops, params)
        if troop:
            deploy_troop(troop, position)
            # Remember what we deployed
            logic.pattern_memory['our_last_deployment'] = (troop, position, current_time)

def update_pattern_memory(opp_troops, current_time):
    \"\"\"Update memory with current opponent troops and positions\"\"\"
    # Skip if no change in game time (same frame)
    if current_time == logic.pattern_memory['last_game_time']:
        return
    
    # Record current troops and positions
    current_troops = [troop.name for troop in opp_troops]
    current_positions = [(troop.name, troop.position) for troop in opp_troops]
    
    # Detect new deployments (compare with last known troops)
    if hasattr(logic, 'last_opponent_troops_snapshot'):
        last_troops_set = set(logic.last_opponent_troops_snapshot)
        current_troops_set = set(current_troops)
        
        # Newly deployed troops
        new_troops = current_troops_set - last_troops_set
        
        if new_troops:
            # Record the deployment time
            logic.pattern_memory['deployment_times'].append(current_time)
            if len(logic.pattern_memory['deployment_times']) > {memory_length}:
                logic.pattern_memory['deployment_times'].pop(0)
            
            # Check if our last deployment triggered this response
            if logic.pattern_memory['our_last_deployment']:
                our_troop, our_pos, our_time = logic.pattern_memory['our_last_deployment']
                response_time = current_time - our_time
                
                # Only count as response if within reasonable time
                if 10 <= response_time <= 150:
                    # Record the response
                    if our_troop not in logic.pattern_memory['opponent_responses']:
                        logic.pattern_memory['opponent_responses'][our_troop] = []
                    
                    logic.pattern_memory['opponent_responses'][our_troop].append(
                        (list(new_troops), response_time)
                    )
                    
                    # Keep last {memory_length} responses
                    if len(logic.pattern_memory['opponent_responses'][our_troop]) > {memory_length}:
                        logic.pattern_memory['opponent_responses'][our_troop].pop(0)
    
    # Update troop and position history
    logic.pattern_memory['last_opponent_troops'].append(current_troops)
    if len(logic.pattern_memory['last_opponent_troops']) > {memory_length}:
        logic.pattern_memory['last_opponent_troops'].pop(0)
    
    logic.pattern_memory['last_opponent_positions'].append(current_positions)
    if len(logic.pattern_memory['last_opponent_positions']) > {memory_length}:
        logic.pattern_memory['last_opponent_positions'].pop(0)
    
    # Update current time and troops snapshot
    logic.pattern_memory['last_game_time'] = current_time
    logic.last_opponent_troops_snapshot = current_troops

def analyze_patterns():
    \"\"\"Analyze opponent patterns from recorded memory\"\"\"
    # Skip analysis if we don't have enough data
    if len(logic.pattern_memory['last_opponent_troops']) < 3:
        return
    
    # Analyze deployment timings
    if len(logic.pattern_memory['deployment_times']) >= 2:
        intervals = [logic.pattern_memory['deployment_times'][i] - logic.pattern_memory['deployment_times'][i-1] 
                    for i in range(1, len(logic.pattern_memory['deployment_times']))]
        
        # Detect regular timing patterns
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            # Store this as a recognized pattern
            logic.pattern_memory['repeat_patterns']['deployment_timing'] = avg_interval
    
    # Analyze lane preferences
    lane_counts = {{"left": 0, "center": 0, "right": 0}}
    
    for positions in logic.pattern_memory['last_opponent_positions']:
        for _, pos in positions:
            x = pos[0]
            if x < -10:
                lane_counts["left"] += 1
            elif x > 10:
                lane_counts["right"] += 1
            else:
                lane_counts["center"] += 1
    
    # Determine preferred lane if there's a clear preference
    total_positions = sum(lane_counts.values())
    if total_positions > 0:
        preferred_lane = max(lane_counts.items(), key=lambda x: x[1])
        if preferred_lane[1] / total_positions > 0.6:  # Clear preference (>60%)
            logic.pattern_memory['repeat_patterns']['preferred_lane'] = preferred_lane[0]
    
    # Analyze troop type preferences
    troop_counts = {{}}
    for troop_list in logic.pattern_memory['last_opponent_troops']:
        for troop in troop_list:
            if troop not in troop_counts:
                troop_counts[troop] = 0
            troop_counts[troop] += 1
    
    if troop_counts:
        # Get most frequently used troops
        sorted_troops = sorted(troop_counts.items(), key=lambda x: x[1], reverse=True)
        logic.pattern_memory['repeat_patterns']['favorite_troops'] = [t[0] for t in sorted_troops[:3]]

def parse_signal(signal):
    \"\"\"Parse strategy parameters from team_signal\"\"\"
    # Default parameters
    params = {{
        "lane": "{params.get('lane_preference', 'adaptive')}",
        "min_elixir": {params.get('elixir_thresholds', {}).get('min_deploy', 4)},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)},
        "memory_length": {memory_length},
        "adaptation_rate": {adaptation_rate}
    }}
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    \"\"\"Track opponent troop types in team_signal\"\"\"
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def should_deploy_pattern_based(my_tower, opp_tower, opp_troops, params):
    \"\"\"Decide whether to deploy troops based on patterns and current game state\"\"\"
    # Check if we have enough elixir
    if my_tower.total_elixir < params["min_elixir"]:
        return False
    
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
    
    # Get current game time
    current_time = my_tower.game_timer
    
    # If we know opponent's deployment timing, try to deploy right before they do
    if 'deployment_timing' in logic.pattern_memory['repeat_patterns']:
        avg_interval = logic.pattern_memory['repeat_patterns']['deployment_timing']
        
        # If we have deployment times, check if it's close to next expected deployment
        if logic.pattern_memory['deployment_times']:
            last_deploy = logic.pattern_memory['deployment_times'][-1]
            next_expected = last_deploy + avg_interval
            
            # If next deployment is expected soon, deploy now to counter
            if current_time >= next_expected - 30 and current_time <= next_expected:
                if my_tower.total_elixir >= params["min_elixir"] + 1:
                    return True
    
    # If opponent has predictable responses to our troops, consider that
    if logic.pattern_memory['our_last_deployment']:
        our_troop, _, our_time = logic.pattern_memory['our_last_deployment']
        time_since_deploy = current_time - our_time
        
        # If we recently deployed and expect a counter, wait a bit
        if time_since_deploy < 30 and our_troop in logic.pattern_memory['opponent_responses']:
            responses = logic.pattern_memory['opponent_responses'][our_troop]
            if responses and any(r[1] < 60 for r in responses):  # Quick responses
                # Wait for their counter unless we have lots of elixir
                if my_tower.total_elixir < 8:
                    return False
    
    # Deploy based on elixir availability
    if my_tower.total_elixir >= 8:  # Lots of elixir, deploy something
        return True
    elif my_tower.total_elixir >= params["min_elixir"] + 2:  # Extra elixir available
        return random.random() > 0.3  # 70% chance to deploy
    
    # Default: deploy if we meet minimum elixir
    return my_tower.total_elixir >= params["min_elixir"]

def choose_counter_pattern(my_tower, opp_troops, params):
    \"\"\"Choose the best troop and position to counter observed patterns\"\"\"
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Base troop scores
    troop_scores = {{
        "Wizard": 6 if "Wizard" in available_troops else 0,
        "Dragon": 7 if "Dragon" in available_troops else 0,
        "Musketeer": 5 if "Musketeer" in available_troops else 0,
        "Valkyrie": 6 if "Valkyrie" in available_troops else 0,
        "Knight": 4 if "Knight" in available_troops else 0,
        "Archer": 3 if "Archer" in available_troops else 0,
        "Barbarian": 2 if "Barbarian" in available_troops else 0,
        "Minion": 3 if "Minion" in available_troops else 0
    }}
    
    # Counter patterns if we have identified any
    patterns = logic.pattern_memory['repeat_patterns']
    
    # Counter favorite troops if we know them
    if 'favorite_troops' in patterns:
        favorite_troops = patterns['favorite_troops']
        
        for fav_troop in favorite_troops:
            # Add counters for specific troops
            if fav_troop == "Dragon":
                troop_scores["Musketeer"] += 4
                troop_scores["Archer"] += 2
            elif fav_troop in ["Knight", "Barbarian", "Valkyrie"]:
                troop_scores["Dragon"] += 3
                troop_scores["Valkyrie"] += 3
                troop_scores["Wizard"] += 2
            elif fav_troop in ["Wizard", "Archer", "Musketeer"]:
                troop_scores["Knight"] += 3
                troop_scores["Dragon"] += 2
    
    # Adjust scores based on opponent's lane preference
    if 'preferred_lane' in patterns:
        preferred_lane = patterns['preferred_lane']
        # We'll use this later for positioning
    
    # If opponent makes predictable responses to our troops, use that
    if logic.pattern_memory['opponent_responses']:
        # Find the troop that gets the weakest counter from opponent
        best_response_score = -1
        best_response_troop = None
        
        for our_troop, responses in logic.pattern_memory['opponent_responses'].items():
            if our_troop in available_troops:
                # Calculate how threatening their responses are (lower is better for us)
                threat_level = 0
                for troops_list, response_time in responses:
                    # Quick responses are more threatening
                    time_factor = 1.0 if response_time < 30 else 0.7
                    # More troops in response is more threatening
                    troop_count_factor = len(troops_list) * 0.8
                    threat_level += time_factor * troop_count_factor
                
                # Average threat level across responses
                if responses:
                    avg_threat = threat_level / len(responses)
                    response_score = 10 - avg_threat  # Invert so higher is better
                    
                    if response_score > best_response_score:
                        best_response_score = response_score
                        best_response_troop = our_troop
        
        # Boost score for troop with best response pattern
        if best_response_troop and best_response_score > 5:
            troop_scores[best_response_troop] += 4
    
    # Choose the highest scoring available troop
    best_score = -1
    chosen_troop = None
    
    for troop, score in troop_scores.items():
        if score > best_score and troop in available_troops:
            best_score = score
            chosen_troop = troop
    
    # Fallback to first available troop if none selected
    if not chosen_troop and available_troops:
        chosen_troop = available_troops[0]
    
    # Determine position based on patterns
    lane = params["lane"]
    
    # If opponent has a preferred lane, counter it
    if 'preferred_lane' in patterns and not defensive_mode:
        opp_lane = patterns['preferred_lane']
        if opp_lane == "left":
            lane = "right"  # Attack opposite lane
        elif opp_lane == "right":
            lane = "left"  # Attack opposite lane
        else:
            # If they prefer center, attack from sides
            lane = random.choice(["left", "right"])
    elif defensive_mode and 'preferred_lane' in patterns:
        # In defensive mode, match opponent's lane
        lane = patterns['preferred_lane']
    
    # Set deployment position based on lane
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right
        if not hasattr(choose_counter_pattern, 'last_lane'):
            choose_counter_pattern.last_lane = "left"
        
        if choose_counter_pattern.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            choose_counter_pattern.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            choose_counter_pattern.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on defensive mode
    y_pos = params["y_defensive"] if defensive_mode else params["y_default"]
    
    return chosen_troop, (x_pos, y_pos)

def deploy_troop(troop, position):
    \"\"\"Deploy the selected troop at the given position\"\"\"
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code
    
    @staticmethod
    def troop_synergy_template(params: Dict[str, Any]) -> str:
        """
        Generate a troop synergy strategy template that deploys complementary
        troops that work well together.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        
        code = f"""from teams.helper_function import Troops, Utils
import random

team_name = "{params.get('name', 'Troop_Synergy')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{params.get('lane_preference', 'center')[0]},e:{params.get('elixir_thresholds', {}).get('min_deploy', 4):.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:synergy"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Initialize synergy tracking
    if not hasattr(logic, 'synergy_data'):
        logic.synergy_data = {{
            'deployed_combos': [],  # List of deployed troop combinations
            'combo_effectiveness': {{}},  # Effectiveness of each combo
            'last_hp_check': my_tower.health,  # Tower health at last check
            'last_deployment_time': 0,  # When we last deployed
            'active_synergy': None,  # Current synergy strategy in use
            'synergy_start_time': 0,  # When we started current synergy
            'recent_deployments': []  # Recent troops deployed
        }}
    
    # Update synergy effectiveness data
    update_synergy_data(my_tower, my_troops)
    
    # Analyze whether to deploy troops based on elixir and game state
    if should_deploy(my_tower, opp_tower, opp_troops, params):
        # Choose best troop and position based on synergy
        troop, position = choose_synergy_troop(my_tower, my_troops, opp_troops, params)
        if troop:
            deploy_troop(troop, position)
            
            # Track deployment
            logic.synergy_data['last_deployment_time'] = my_tower.game_timer
            logic.synergy_data['recent_deployments'].append((troop, my_tower.game_timer))
            
            # Clean up old deployments
            current_time = my_tower.game_timer
            logic.synergy_data['recent_deployments'] = [
                (t, time) for t, time in logic.synergy_data['recent_deployments'] 
                if current_time - time < 300  # Keep last 5 seconds
            ]

def update_synergy_data(my_tower, my_troops):
    \"\"\"Update data about synergy effectiveness\"\"\"
    # Check if our troops are successfully working together
    current_hp = my_tower.health
    if hasattr(logic, 'last_hp_check'):
        # If we've taken damage, our synergy might not be working well
        hp_change = current_hp - logic.synergy_data['last_hp_check']
        
        # Only evaluate if we have an active synergy
        if logic.synergy_data['active_synergy'] and hp_change < 0:
            synergy = logic.synergy_data['active_synergy']
            # Record negative effectiveness
            if synergy not in logic.synergy_data['combo_effectiveness']:
                logic.synergy_data['combo_effectiveness'][synergy] = 0
            # Decrease effectiveness score
            logic.synergy_data['combo_effectiveness'][synergy] -= min(abs(hp_change) / 500, 0.5)
    
    # Evaluate troops on the field to detect effective combinations
    troop_types = [troop.name for troop in my_troops]
    
    # Check for specific synergy combinations
    combinations = [
        ("Wizard", "Knight"),  # Ranged + Tank
        ("Dragon", "Valkyrie"),  # Air + Ground splash
        ("Musketeer", "Barbarian"),  # Ranged + Swarm
        ("Archer", "Dragon"),  # Multi-target ranged
        ("Valkyrie", "Musketeer")  # Splash + Single target
    ]
    
    for combo in combinations:
        if all(t in troop_types for t in combo):
            combo_key = f"{{combo[0]}}_{{combo[1]}}"
            
            # If this combo isn't active, mark it as active
            if logic.synergy_data['active_synergy'] != combo_key:
                logic.synergy_data['active_synergy'] = combo_key
                logic.synergy_data['synergy_start_time'] = my_tower.game_timer
            
            # If we have this combo and haven't taken damage, it might be working
            else:
                duration = my_tower.game_timer - logic.synergy_data['synergy_start_time']
                if duration > 100:  # If combo lasted over 100 frames
                    # Record positive effectiveness
                    if combo_key not in logic.synergy_data['combo_effectiveness']:
                        logic.synergy_data['combo_effectiveness'][combo_key] = 0
                    # Increase effectiveness score
                    logic.synergy_data['combo_effectiveness'][combo_key] += min(duration / 1000, 0.5)
    
    # Track the HP for next check
        logic.synergy_data['last_hp_check'] = current_hp

def parse_signal(signal):
    '''Parse strategy parameters from team_signal'''
    # Default parameters
    params = {
        "lane": "{params.get('lane_preference', 'center')}",
        "min_elixir": {params.get('elixir_thresholds', {}).get('min_deploy', 4)},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)}
    }
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    '''Track opponent troop types in team_signal'''
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def should_deploy(my_tower, opp_tower, opp_troops, params):
    '''Decide whether to deploy troops based on strategy and game state'''
    # Check if we have enough elixir
    if my_tower.total_elixir < params["min_elixir"]:
        return False
    
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
    
    # Always deploy early in the game
    if my_tower.game_timer < 300:
        return True
    
    # Deploy based on elixir availability
    if my_tower.total_elixir >= 8:  # Lots of elixir, deploy something
        return True
    elif my_tower.total_elixir >= params["min_elixir"] + 2:  # Extra elixir available
        return random.random() > 0.3  # 70% chance to deploy
    
    # Default: deploy if we meet minimum elixir
    return my_tower.total_elixir >= params["min_elixir"]

def choose_synergy_troop(my_tower, my_troops, opp_troops, params):
    '''Choose the best troop to deploy for effective synergy combinations'''
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Initialize synergy pairs with their scores
    synergy_pairs = {
        # Tank + Damage dealer pairs
        "Knight_Wizard": 9,
        "Knight_Musketeer": 8,
        "Knight_Archer": 7,
        
        # Air + Ground combinations
        "Dragon_Valkyrie": 10,
        "Dragon_Knight": 8,
        "Minion_Barbarian": 7,
        
        # Splash + Single-target damage
        "Wizard_Musketeer": 9,
        "Valkyrie_Archer": 8,
        "Dragon_Archer": 8,
        
        # Defensive combinations
        "Valkyrie_Knight": 7,
        "Wizard_Barbarian": 8,
        "Musketeer_Minion": 7
    }
    
    # Look at our current troops to find missing synergy components
    current_troops = [troop.name for troop in my_troops]
    
    # Initialize troop scores based on individual value
    troop_scores = {
        "Wizard": 6 if "Wizard" in available_troops else 0,
        "Dragon": 7 if "Dragon" in available_troops else 0,
        "Musketeer": 5 if "Musketeer" in available_troops else 0,
        "Valkyrie": 6 if "Valkyrie" in available_troops else 0,
        "Knight": 4 if "Knight" in available_troops else 0,
        "Archer": 3 if "Archer" in available_troops else 0,
        "Barbarian": 2 if "Barbarian" in available_troops else 0,
        "Minion": 3 if "Minion" in available_troops else 0
    }
    
    # Adjust scores based on completing synergies with troops we already have
    for troop_name in current_troops:
        for synergy_key, score in synergy_pairs.items():
            parts = synergy_key.split("_")
            if troop_name in parts:
                # Find the complementary troop in the pair
                complementary_troop = parts[0] if parts[1] == troop_name else parts[1]
                
                # Check if we've learned this combo is effective
                effectiveness = 0
                if hasattr(logic, 'synergy_data'):
                    effectiveness = logic.synergy_data['combo_effectiveness'].get(synergy_key, 0)
                
                # Boost score for the complementary troop
                if complementary_troop in available_troops:
                    # Base boost plus any learned effectiveness
                    boost = min(score * 0.5, 5) + min(effectiveness * 2, 5)
                    troop_scores[complementary_troop] += boost
    
    # Adjust scores based on opponent troops
    if opp_troops:
        air_count = sum(1 for troop in opp_troops if troop.type == "air")
        ground_count = sum(1 for troop in opp_troops if troop.type == "ground")
        
        if air_count > ground_count:
            # Prioritize anti-air troops
            troop_scores["Wizard"] += 3
            troop_scores["Musketeer"] += 3
            troop_scores["Archer"] += 2
        else:
            # Prioritize anti-ground troops
            troop_scores["Valkyrie"] += 3
            troop_scores["Knight"] += 2
            troop_scores["Barbarian"] += 2
    
    # Choose the highest scoring available troop
    best_score = -1
    chosen_troop = None
    
    for troop, score in troop_scores.items():
        if score > best_score and troop in available_troops:
            best_score = score
            chosen_troop = troop
    
    # Fallback to first available troop if none selected
    if not chosen_troop and available_troops:
        chosen_troop = available_troops[0]
    
    # Determine position based on lane preference and synergies
    lane = params["lane"]
    
    # If we already have troops deployed, try to deploy near them for better synergy
    position_bias = None
    if current_troops and my_troops:
        # Find the average position of our troops
        x_sum = sum(troop.position[0] for troop in my_troops)
        x_avg = x_sum / len(my_troops)
        
        # Bias our deployment towards existing troops
        if x_avg < -10:
            lane = "left"
        elif x_avg > 10:
            lane = "right"
        else:
            lane = "center"
    
    # Set deployment position based on lane
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right for split pushing
        if not hasattr(choose_synergy_troop, 'last_lane'):
            choose_synergy_troop.last_lane = "left"
        
        if choose_synergy_troop.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            choose_synergy_troop.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            choose_synergy_troop.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on defensive mode
    y_pos = params["y_defensive"] if defensive_mode else params["y_default"]
    
    return chosen_troop, (x_pos, y_pos)

def deploy_troop(troop, position):
    '''Deploy the selected troop at the given position'''
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code
    
    @staticmethod
    def fully_adaptive_template(params: Dict[str, Any]) -> str:
        """
        Generate a fully adaptive strategy template that combines all adaptive
        techniques for maximum responsiveness.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        # Extract parameters from all systems
        counter_weights = params.get("counter_weights", {})
        air_vs_ground = counter_weights.get("air_vs_ground", 1.2)
        
        elixir_thresholds = params.get("elixir_thresholds", {})
        min_deploy = elixir_thresholds.get("min_deploy", 4)
        
        phase_params = params.get("phase_parameters", {})
        early_game_threshold = phase_params.get("early_game_threshold", 500)
        late_game_threshold = phase_params.get("late_game_threshold", 1200)
        
        memory_params = params.get("memory_parameters", {})
        memory_length = memory_params.get("memory_length", 5)
        
        code = f"""from teams.helper_function import Troops, Utils
import random
import math

team_name = "{params.get('name', 'Adaptive_Master')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{params.get('lane_preference', 'adaptive')[0]},e:{min_deploy:.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:adaptive"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Initialize adaptive memory system
    if not hasattr(logic, 'adaptive_memory'):
        logic.adaptive_memory = {{
            # Pattern recognition
            'opponent_patterns': {{
                'troop_preferences': {{}},
                'lane_preferences': {{'left': 0, 'center': 0, 'right': 0}},
                'deployment_times': [],
                'response_patterns': {{}}
            }},
            # Phase tracking
            'current_phase': 'early',
            'last_phase_change': 0,
            # Elixir management
            'elixir_history': [],
            'opponent_spending_rate': 0,
            'last_deployment': None,
            # Synergy tracking
            'effective_combos': {{}},
            'current_troops': [],
            # Game state
            'last_hp': 0,
            'damage_taken': 0,
            'successful_defenses': 0,
            'opponent_adaptations': 0
        }}
    
    # Determine current game phase
    current_phase = determine_game_phase(my_tower)
    
    # Update adaptive memory with current game state
    update_adaptive_memory(my_tower, opp_tower, my_troops, opp_troops, current_phase)
    
    # Decide which adaptive strategy to use based on game state
    strategy = choose_adaptive_strategy(my_tower, opp_tower, current_phase)
    
    # Determine if we should deploy troops based on chosen strategy
    if should_deploy_adaptive(my_tower, opp_tower, opp_troops, params, strategy):
        # Choose best troop and position using the selected strategy
        troop, position = select_troop_and_position(my_tower, my_troops, opp_troops, params, strategy)
        if troop:
            deploy_troop(troop, position)
            # Record this deployment
            logic.adaptive_memory['last_deployment'] = (troop, position, my_tower.game_timer)

def determine_game_phase(my_tower):
    \"\"\"Determine the current game phase based on game timer and state\"\"\"
    game_timer = my_tower.game_timer
    
    # Basic phase determination by timer
    if game_timer < {early_game_threshold}:
        current_phase = "early"
    elif game_timer < {late_game_threshold}:
        current_phase = "mid"
    else:
        current_phase = "late"
    
    # Store phase transition if it changed
    if current_phase != logic.adaptive_memory['current_phase']:
        logic.adaptive_memory['last_phase_change'] = game_timer
        logic.adaptive_memory['current_phase'] = current_phase
    
    return current_phase

def update_adaptive_memory(my_tower, opp_tower, my_troops, opp_troops, current_phase):
    \"\"\"Update adaptive memory with current game state information\"\"\"
    # Track elixir usage
    logic.adaptive_memory['elixir_history'].append(my_tower.total_elixir)
    if len(logic.adaptive_memory['elixir_history']) > 100:  # Keep reasonable history
        logic.adaptive_memory['elixir_history'].pop(0)
    
    # Track opponent troop preferences
    for troop in opp_troops:
        if troop.name not in logic.adaptive_memory['opponent_patterns']['troop_preferences']:
            logic.adaptive_memory['opponent_patterns']['troop_preferences'][troop.name] = 0
        logic.adaptive_memory['opponent_patterns']['troop_preferences'][troop.name] += 1
    
    # Track opponent lane preferences
    for troop in opp_troops:
        x_pos = troop.position[0]
        if x_pos < -10:
            logic.adaptive_memory['opponent_patterns']['lane_preferences']['left'] += 1
        elif x_pos > 10:
            logic.adaptive_memory['opponent_patterns']['lane_preferences']['right'] += 1
        else:
            logic.adaptive_memory['opponent_patterns']['lane_preferences']['center'] += 1
    
    # Track damage taken for defense effectiveness
    current_hp = my_tower.health
    if logic.adaptive_memory['last_hp'] > 0:
        hp_change = current_hp - logic.adaptive_memory['last_hp']
        if hp_change < 0:
            logic.adaptive_memory['damage_taken'] -= hp_change  # Store as positive value
    logic.adaptive_memory['last_hp'] = current_hp
    
    # Track our current troops for synergy
    logic.adaptive_memory['current_troops'] = [troop.name for troop in my_troops]
    
    # Update effective combos if we have multiple troops
    if len(my_troops) >= 2 and logic.adaptive_memory['damage_taken'] == 0:
        troop_names = sorted([troop.name for troop in my_troops])
        for i in range(len(troop_names)):
            for j in range(i+1, len(troop_names)):
                combo = f"{{troop_names[i]}}_{{troop_names[j]}}"
                if combo not in logic.adaptive_memory['effective_combos']:
                    logic.adaptive_memory['effective_combos'][combo] = 0
                logic.adaptive_memory['effective_combos'][combo] += 1

def choose_adaptive_strategy(my_tower, opp_tower, current_phase):
    \"\"\"Choose the most appropriate adaptive strategy for the current game state\"\"\"
    # Default strategy weights
    strategy_weights = {{
        'counter_picker': 1.0,  # Counter opponent troops
        'lane_adaptor': 1.0,    # Adapt to opponent lanes
        'elixir_optimizer': 1.0,  # Optimize elixir efficiency
        'phase_shifter': 1.0,   # Phase-specific strategy
        'pattern_recognition': 1.0,  # Learn from opponent patterns
        'troop_synergy': 1.0    # Deploy complementary troops
    }}
    
    # Adjust weights based on game phase
    if current_phase == "early":
        strategy_weights['elixir_optimizer'] += 0.5  # Early game elixir management is important
        strategy_weights['troop_synergy'] += 0.3    # Build good combos early
    elif current_phase == "mid":
        strategy_weights['counter_picker'] += 0.5   # Mid game countering is important
        strategy_weights['pattern_recognition'] += 0.4  # Pattern recognition starts to help
    else:  # late game
        strategy_weights['lane_adaptor'] += 0.5     # Late game lane pressure
        strategy_weights['phase_shifter'] += 0.7    # Late game aggressive strategy
    
    # Adjust based on tower health
    my_health = my_tower.health / 10000
    opp_health = opp_tower.health / 10000
    
    if my_health < opp_health - 0.2:  # We're losing
        strategy_weights['counter_picker'] += 0.6   # Counter their troops
        strategy_weights['elixir_optimizer'] += 0.4  # Be more efficient
    elif my_health > opp_health + 0.2:  # We're winning
        strategy_weights['troop_synergy'] += 0.5    # Focus on synergy
        strategy_weights['lane_adaptor'] += 0.3     # Apply lane pressure
    
    # Adjust based on past success
    if logic.adaptive_memory['successful_defenses'] > 3:
        strategy_weights['pattern_recognition'] += 0.3  # We're good at recognizing patterns
    
    if logic.adaptive_memory['effective_combos']:
        strategy_weights['troop_synergy'] += 0.3  # We have effective combos
    
    # Choose the highest weighted strategy
    chosen_strategy = max(strategy_weights.items(), key=lambda x: x[1])[0]
    return chosen_strategy

def should_deploy_adaptive(my_tower, opp_tower, opp_troops, params, strategy):
    \"\"\"Decide whether to deploy troops using the chosen adaptive strategy\"\"\"
    # Base conditions that always apply
    if my_tower.total_elixir < params["min_elixir"]:
        return False
    
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
    
    # Strategy-specific deployment decisions
    if strategy == 'elixir_optimizer':
        # Be more conservative with elixir
        if my_tower.total_elixir < params["min_elixir"] + 1:
            return False
        elif my_tower.total_elixir >= 8:  # But deploy if we have lots
            return True
        else:
            return random.random() > 0.5  # 50% chance to hold
            
    elif strategy == 'counter_picker':
        # Deploy when we have enough elixir to counter
        return my_tower.total_elixir >= params["min_elixir"] + 1
        
    elif strategy == 'phase_shifter':
        # Phase-specific deployment thresholds
        phase = logic.adaptive_memory['current_phase']
        if phase == 'early':
            # More conservative early
            return my_tower.total_elixir >= params["min_elixir"] + 1 and random.random() > 0.3
        elif phase == 'mid':
            # Standard in mid-game
            return my_tower.total_elixir >= params["min_elixir"] and random.random() > 0.2
        else:  # late
            # More aggressive late
            return my_tower.total_elixir >= max(params["min_elixir"] - 1, 3)
            
    elif strategy == 'troop_synergy':
        # Deploy if we have a troop on the field and need its synergy partner
        if logic.adaptive_memory['current_troops'] and my_tower.total_elixir >= params["min_elixir"]:
            return True
            
    elif strategy == 'pattern_recognition':
        # Deploy if we're expecting opponent to deploy soon
        last_deployment = logic.adaptive_memory.get('last_deployment')
        if last_deployment:
            _, _, time = last_deployment
            time_since_deploy = my_tower.game_timer - time
            # If we just deployed, hold off
            if time_since_deploy < 30:
                return my_tower.total_elixir >= 8  # Unless we have lots of elixir
    
    # Default case - use base logic
    if my_tower.total_elixir >= 8:  # Lots of elixir
        return True
    elif my_tower.total_elixir >= params["min_elixir"] + 2:  # Extra elixir
        return random.random() > 0.3
    
    # Default
    return my_tower.total_elixir >= params["min_elixir"]

def select_troop_and_position(my_tower, my_troops, opp_troops, params, strategy):
    \"\"\"Select troop and position using the chosen adaptive strategy\"\"\"
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Base troop scores
    troop_scores = {{
        "Wizard": 6 if "Wizard" in available_troops else 0,
        "Dragon": 7 if "Dragon" in available_troops else 0,
        "Musketeer": 5 if "Musketeer" in available_troops else 0,
        "Valkyrie": 6 if "Valkyrie" in available_troops else 0,
        "Knight": 4 if "Knight" in available_troops else 0,
        "Archer": 3 if "Archer" in available_troops else 0,
        "Barbarian": 2 if "Barbarian" in available_troops else 0,
        "Minion": 3 if "Minion" in available_troops else 0
    }}
    
    # Adjust scores based on chosen strategy
    if strategy == 'counter_picker':
        # Count opponent troop types
        air_count = sum(1 for troop in opp_troops if troop.type == "air")
        ground_count = sum(1 for troop in opp_troops if troop.type == "ground")
        
        if air_count > ground_count * {air_vs_ground}:
            # Counter air
            troop_scores["Musketeer"] += 5
            troop_scores["Wizard"] += 4
            troop_scores["Archer"] += 3
        elif ground_count > air_count * {air_vs_ground}:
            # Counter ground
            troop_scores["Dragon"] += 5
            troop_scores["Wizard"] += 4
            troop_scores["Valkyrie"] += 3
            
    elif strategy == 'elixir_optimizer':
        # Calculate efficiency (value/cost)
        troop_costs = {{
            "Wizard": 5,
            "Dragon": 4,
            "Musketeer": 4,
            "Valkyrie": 4,
            "Knight": 3,
            "Archer": 3,
            "Barbarian": 3,
            "Minion": 3
        }}
        
        for troop, base_score in troop_scores.items():
            if troop in troop_costs:
                efficiency = base_score / troop_costs[troop]
                # Boost efficient troops
                troop_scores[troop] = base_score + (efficiency * 3)
                
    elif strategy == 'phase_shifter':
        # Phase-specific troops
        phase = logic.adaptive_memory['current_phase']
        if phase == 'early':
            # Early game preferences
            troop_scores["Knight"] += 3
            troop_scores["Archer"] += 4
        elif phase == 'mid':
            # Mid game preferences
            troop_scores["Valkyrie"] += 4
            troop_scores["Wizard"] += 3
        else:  # late
            # Late game preferences
            troop_scores["Dragon"] += 4
            troop_scores["Wizard"] += 4
            
    elif strategy == 'troop_synergy':
        # Boost scores for troops that synergize with current troops
        current_troops = logic.adaptive_memory['current_troops']
        synergy_pairs = {{
            "Knight_Wizard": 4,
            "Dragon_Valkyrie": 5,
            "Musketeer_Barbarian": 3,
            "Archer_Dragon": 4,
            "Valkyrie_Musketeer": 4
        }}
        
        for troop_name in current_troops:
            for synergy_key, boost in synergy_pairs.items():
                parts = synergy_key.split("_")
                if troop_name in parts:
                    # Find the complementary troop
                    complement = parts[0] if parts[1] == troop_name else parts[1]
                    if complement in troop_scores:
                        troop_scores[complement] += boost
                        
    elif strategy == 'pattern_recognition':
        # Adjust based on observed patterns of opponent responses
        if 'response_patterns' in logic.adaptive_memory['opponent_patterns']:
            responses = logic.adaptive_memory['opponent_patterns']['response_patterns']
            
            # Find troops that have worked well in the past
            for troop, score in troop_scores.items():
                if troop in responses and responses[troop] > 0:
                    troop_scores[troop] += responses[troop] * 2
    
    # Choose the highest scoring available troop
    best_score = -1
    chosen_troop = None
    
    for troop, score in troop_scores.items():
        if score > best_score and troop in available_troops:
            best_score = score
            chosen_troop = troop
    
    # Fallback to first available troop if none selected
    if not chosen_troop and available_troops:
        chosen_troop = available_troops[0]
    
    # Determine position based on strategy and opponent patterns
    lane = params["lane"]
    
    # Dynamic lane selection
    if strategy == 'lane_adaptor' and 'lane_preferences' in logic.adaptive_memory['opponent_patterns']:
        lane_prefs = logic.adaptive_memory['opponent_patterns']['lane_preferences']
        max_lane = max(lane_prefs.items(), key=lambda x: x[1])
        
        # Determine whether to match or counter opponent lane
        defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
        if defensive_mode:
            # Match opponent's lane to defend
            lane = max_lane[0]
        else:
            # Counter opponent's lane
            if max_lane[0] == "left":
                lane = "right"
            elif max_lane[0] == "right":
                lane = "left"
            else:
                # If opponent uses center, split push
                lane = "split"
    
    # Set deployment position based on lane
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right
        if not hasattr(select_troop_and_position, 'last_lane'):
            select_troop_and_position.last_lane = "left"
        
        if select_troop_and_position.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            select_troop_and_position.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            select_troop_and_position.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on strategy and defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    y_base = params["y_defensive"] if defensive_mode else params["y_default"]
    
    # Adjust y position based on strategy
    if strategy == 'phase_shifter':
        # Phase-specific y position
        phase = logic.adaptive_memory['current_phase']
        if phase == 'early':
            y_pos = y_base - 5  # More defensive in early game
        elif phase == 'mid':
            y_pos = y_base  # Normal in mid game
        else:  # late game
            y_pos = y_base + 5  # More aggressive in late game
    else:
        y_pos = y_base
    
    # Ensure y position is within reasonable bounds
    y_pos = max(10, min(y_pos, 50))
    
    return chosen_troop, (x_pos, y_pos)

def parse_signal(signal):
    '''Parse strategy parameters from team_signal'''
    # Default parameters
    params = {
        "lane": "{params.get('lane_preference', 'adaptive')}",
        "min_elixir": {min_deploy},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)}
    }
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    '''Track opponent troop types in team_signal'''
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def deploy_troop(troop, position):
    '''Deploy the selected troop at the given position'''
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code
    
    @staticmethod
    def baseline_template(params: Dict[str, Any]) -> str:
        """
        Generate a baseline strategy template with basic functionality.
        
        Args:
            params: Strategy parameters
            
        Returns:
            Python code string
        """
        code = f"""from teams.helper_function import Troops, Utils
import random

team_name = "{params.get('name', 'Baseline')}"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:{params.get('lane_preference', 'center')[0]},e:{params.get('elixir_thresholds', {}).get('min_deploy', 4):.1f},d:{params.get('defensive_trigger', 0.6):.1f},y:{params.get('position_settings', {}).get('y_default', 40)},t:basic"

def random_x(min_val={params.get('position_settings', {}).get('x_range', [-20, 20])[0]}, max_val={params.get('position_settings', {}).get('x_range', [-20, 20])[1]}):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    \"\"\"
    DON'T TEMPER DEPLOY FUNCTION
    \"\"\"
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Parse strategy parameters from team_signal
    params = parse_signal(team_signal)
    
    # Track opponent troops in team_signal
    track_opponent_troops(opp_troops)
    
    # Analyze whether to deploy troops based on elixir and game state
    if should_deploy(my_tower, opp_tower, opp_troops, params):
        # Choose best troop and position based on strategy
        troop, position = choose_troop_and_position(my_tower, opp_troops, params)
        if troop:
            deploy_troop(troop, position)

def parse_signal(signal):
    \"\"\"Parse strategy parameters from team_signal\"\"\"
    # Default parameters
    params = {{
        "lane": "{params.get('lane_preference', 'center')}",
        "min_elixir": {params.get('elixir_thresholds', {}).get('min_deploy', 4)},
        "defensive_trigger": {params.get('defensive_trigger', 0.6)},
        "y_default": {params.get('position_settings', {}).get('y_default', 40)},
        "y_defensive": {params.get('position_settings', {}).get('defensive_y', 20)}
    }}
    
    # Parse signal if it contains parameters
    if "v:" in signal:
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "l":  # lane preference
                params["lane"] = value
            elif key == "e":  # min elixir
                try:
                    params["min_elixir"] = float(value)
                except ValueError:
                    pass
            elif key == "d":  # defensive trigger
                try:
                    params["defensive_trigger"] = float(value)
                except ValueError:
                    pass
            elif key == "y":  # y position
                try:
                    params["y_default"] = int(value)
                except ValueError:
                    pass
    
    return params

def track_opponent_troops(opp_troops):
    \"\"\"Track opponent troop types in team_signal\"\"\"
    global team_signal
    
    # Extract existing tracked troops
    tracked = set()
    if "," in team_signal:
        parts = team_signal.split(",")
        for part in parts:
            if ":" not in part and part.strip():
                tracked.add(part.strip())
    
    # Add new opponent troop names
    for troop in opp_troops:
        if troop.name not in tracked:
            tracked.add(troop.name)
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += " " + troop.name
    
    # Ensure team_signal doesn't exceed length limit
    if len(team_signal) > 200:
        team_signal = team_signal[:197] + "..."

def should_deploy(my_tower, opp_tower, opp_troops, params):
    \"\"\"Decide whether to deploy troops based on strategy and game state\"\"\"
    # Check if we have enough elixir
    if my_tower.total_elixir < params["min_elixir"]:
        return False
    
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
    
    # Always deploy early in the game
    if my_tower.game_timer < 300:
        return True
    
    # Deploy based on elixir availability
    if my_tower.total_elixir >= 8:  # Lots of elixir, deploy something
        return True
    elif my_tower.total_elixir >= params["min_elixir"] + 2:  # Extra elixir available
        return random.random() > 0.3  # 70% chance to deploy
    
    # Default: deploy if we meet minimum elixir
    return my_tower.total_elixir >= params["min_elixir"]

def choose_troop_and_position(my_tower, opp_troops, params):
    \"\"\"Choose the best troop to deploy and the position\"\"\"
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Choose troop based on strategy
    chosen_troop = None
    
    # Simple heuristic: choose the first available highest-damage troop
    troop_scores = {{
        "Wizard": 10 if "Wizard" in available_troops else 0,
        "Dragon": 9 if "Dragon" in available_troops else 0,
        "Musketeer": 8 if "Musketeer" in available_troops else 0,
        "Valkyrie": 7 if "Valkyrie" in available_troops else 0,
        "Knight": 6 if "Knight" in available_troops else 0,
        "Archer": 5 if "Archer" in available_troops else 0,
        "Barbarian": 4 if "Barbarian" in available_troops else 0,
        "Minion": 3 if "Minion" in available_troops else 0
    }}
    
    # Adjust scores based on opponent troops
    if opp_troops:
        air_count = sum(1 for troop in opp_troops if troop.type == "air")
        ground_count = sum(1 for troop in opp_troops if troop.type == "ground")
        
        if air_count > ground_count:
            # Prioritize anti-air troops
            troop_scores["Wizard"] += 3
            troop_scores["Musketeer"] += 3
            troop_scores["Archer"] += 2
        else:
            # Prioritize anti-ground troops
            troop_scores["Valkyrie"] += 3
            troop_scores["Knight"] += 2
            troop_scores["Barbarian"] += 2
    
    # Choose the highest scoring available troop
    best_score = -1
    for troop, score in troop_scores.items():
        if score > best_score and troop in available_troops:
            best_score = score
            chosen_troop = troop
    
    # Fallback to first available troop if none selected
    if not chosen_troop and available_troops:
        chosen_troop = available_troops[0]
    
    # Determine position based on lane preference
    lane = params["lane"]
    
    # Set x position based on lane preference
    if lane == "left":
        x_pos = random_x(-25, -5)
    elif lane == "right":
        x_pos = random_x(5, 25)
    elif lane == "center":
        x_pos = random_x(-10, 10)
    elif lane == "split":
        # Alternate between left and right for split pushing
        if not hasattr(choose_troop_and_position, 'last_lane'):
            choose_troop_and_position.last_lane = "left"
        
        if choose_troop_and_position.last_lane == "left":
            x_pos = random_x(5, 25)  # Right lane
            choose_troop_and_position.last_lane = "right"
        else:
            x_pos = random_x(-25, -5)  # Left lane
            choose_troop_and_position.last_lane = "left"
    else:
        # Default to random position
        x_pos = random_x()
    
    # Set y position based on defensive mode
    y_pos = params["y_defensive"] if defensive_mode else params["y_default"]
    
    return chosen_troop, (x_pos, y_pos)

def deploy_troop(troop, position):
    \"\"\"Deploy the selected troop at the given position\"\"\"
    if troop == "Archer":
        deploy_list.deploy_archer(position)
    elif troop == "Giant":
        deploy_list.deploy_giant(position)
    elif troop == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop == "Balloon":
        deploy_list.deploy_balloon(position)
    elif troop == "Prince":
        deploy_list.deploy_prince(position)
    elif troop == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop == "Knight":
        deploy_list.deploy_knight(position)
    elif troop == "Minion":
        deploy_list.deploy_minion(position)
    elif troop == "Skeleton":
        deploy_list.deploy_skeleton(position)
    elif troop == "Wizard":
        deploy_list.deploy_wizard(position)
    elif troop == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop == "Musketeer":
        deploy_list.deploy_musketeer(position)
"""
        return code