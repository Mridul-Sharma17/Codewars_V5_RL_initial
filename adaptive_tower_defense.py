from teams.helper_function import Troops, Utils
import random
import math

team_name = "UnstoppableForce"
troops = [
    Troops.dragon, Troops.wizard, Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:a,e:3.75,s:7.5,dt:0.65,y:40,xl:-20,xr:20,tp:dwvmkab,ca:1.5,cs:2.0,ct:1.3"

# ------- Strategy Parameters -------
# These will be optimized by the genetic algorithm
PARAMS = {
    # Elixir management
    "MIN_DEPLOY_ELIXIR": 3.75,     # Minimum elixir to deploy troops
    "SAVE_THRESHOLD": 7.5,         # Save elixir above this amount for opportunities
    "EMERGENCY_THRESHOLD": 2.0,    # Deploy at this threshold when under attack
    
    # Position settings
    "X_RANGE_LEFT": -20,           # Left x-range for deployment
    "X_RANGE_RIGHT": 20,           # Right x-range for deployment
    "Y_DEFAULT": 40,               # Default y position
    "Y_DEFENSIVE": 20,             # Defensive y position (closer to our tower)
    "Y_AGGRESSIVE": 47,            # Aggressive y position (closer to enemy)
    
    # Strategy thresholds
    "DEFENSIVE_TRIGGER": 0.65,     # Tower health ratio to switch to defensive mode
    "AGGRESSION_TRIGGER": 0.25,    # Enemy tower health ratio to trigger aggressive push
    
    # Troop selection weights
    "AIR_VS_GROUND_BONUS": 1.5,    # Bonus for air troops against ground-heavy opponents
    "SPLASH_VS_GROUP_BONUS": 2.0,  # Bonus for splash damage against grouped troops
    "TANK_PRIORITY_BONUS": 1.3,    # Bonus for tanks in priority deployment
    "RANGE_BONUS": 1.5             # Bonus for ranged troops
}

# Troop data for intelligent selection
TROOP_DATA = {
    "Dragon": {
        "elixir": 4, "type": "air", "attack": "splash", "targets": ["air", "ground"],
        "health": 1267, "damage": 176, "range": "medium", "splash": True
    },
    "Wizard": {
        "elixir": 5, "type": "ground", "attack": "splash", "targets": ["air", "ground"],
        "health": 1100, "damage": 410, "range": "high", "splash": True
    },
    "Valkyrie": {
        "elixir": 4, "type": "ground", "attack": "melee", "targets": ["ground"],
        "health": 2097, "damage": 195, "range": "melee", "splash": True
    },
    "Musketeer": {
        "elixir": 4, "type": "ground", "attack": "single", "targets": ["air", "ground"],
        "health": 792, "damage": 239, "range": "high", "splash": False
    },
    "Knight": {
        "elixir": 3, "type": "ground", "attack": "melee", "targets": ["ground"],
        "health": 1938, "damage": 221, "range": "melee", "splash": False
    },
    "Archer": {
        "elixir": 3, "type": "ground", "attack": "single", "targets": ["air", "ground"],
        "health": 334, "damage": 118, "range": "medium", "splash": False
    },
    "Minion": {
        "elixir": 3, "type": "air", "attack": "single", "targets": ["air", "ground"],
        "health": 252, "damage": 129, "range": "short", "splash": False
    },
    "Barbarian": {
        "elixir": 3, "type": "ground", "attack": "melee", "targets": ["ground"],
        "health": 736, "damage": 161, "range": "melee", "splash": False
    }
}

# ------- Match State Tracking -------
match_state = {
    "opponent_troops_seen": set(),    # Track opponent's troops
    "air_troops_count": 0,            # Count of air troops seen
    "ground_troops_count": 0,         # Count of ground troops seen
    "avg_opponent_deploy_time": 0,    # Average time between opponent deployments
    "opponent_deploy_count": 0,       # Count of opponent deployments
    "last_opponent_deploy_time": 0,   # Last time opponent deployed troops
    "total_elixir_spent": 0,          # Total elixir we've spent
    "total_elixir_collected": 0,      # Estimation of total elixir collected
    "game_phase": "early",            # early, mid, late
    "our_tower_health_history": [],   # Track our tower health
    "opponent_adapting": False,       # Flag if opponent seems to be adapting
    "last_troop_deployed": None,      # Last troop we deployed
    "deploy_positions_used": []       # Track our deployment positions
}

def random_x(min_val=None, max_val=None):
    """Generate random x position within specified range"""
    if min_val is None:
        min_val = PARAMS["X_RANGE_LEFT"]
    if max_val is None:
        max_val = PARAMS["X_RANGE_RIGHT"]
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    """
    DON'T TEMPER DEPLOY FUNCTION
    """
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    """Main logic function that coordinates strategy"""
    global team_signal, PARAMS, match_state
    
    # Extract game data
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Update game state 
    update_game_state(my_tower, opp_tower, my_troops, opp_troops)
    
    # Parse strategy parameters from team_signal if needed
    PARAMS = update_params_from_signal(team_signal, PARAMS)
    
    # Determine whether to deploy troops
    if should_deploy_troops(my_tower, opp_tower, my_troops, opp_troops):
        # Choose best troop and position based on current game state
        troop, position = select_best_troop_and_position(my_tower, opp_tower, my_troops, opp_troops)
        
        # Deploy if we found a suitable troop
        if troop:
            deploy_troop(troop, position)
            # Track elixir spent for efficiency metrics
            match_state["total_elixir_spent"] += TROOP_DATA[troop]["elixir"]
            match_state["last_troop_deployed"] = troop
            match_state["deploy_positions_used"].append(position)

def update_game_state(my_tower, opp_tower, my_troops, opp_troops):
    """Update match state tracking information"""
    global match_state
    
    # Track our tower health
    match_state["our_tower_health_history"].append(my_tower.health)
    
    # Track game phase based on game timer
    if my_tower.game_timer < 600:  # First 10 seconds
        match_state["game_phase"] = "early"
    elif my_tower.game_timer < 1200:  # 10-20 seconds
        match_state["game_phase"] = "mid"
    else:  # 20+ seconds
        match_state["game_phase"] = "late"
    
    # Track opponent's troops
    for troop in opp_troops:
        match_state["opponent_troops_seen"].add(troop.name)
        
        # Track when opponent deploys
        if match_state["last_opponent_deploy_time"] != my_tower.game_timer:
            match_state["opponent_deploy_count"] += 1
            deploy_interval = my_tower.game_timer - match_state["last_opponent_deploy_time"]
            
            if match_state["last_opponent_deploy_time"] > 0 and deploy_interval > 0:
                # Update average deploy time
                alpha = 0.3  # Weighting for most recent observations
                match_state["avg_opponent_deploy_time"] = (
                    alpha * deploy_interval + 
                    (1 - alpha) * match_state["avg_opponent_deploy_time"]
                )
            
            match_state["last_opponent_deploy_time"] = my_tower.game_timer
        
        # Count air/ground troops
        if troop.name in TROOP_DATA and TROOP_DATA[troop.name]["type"] == "air":
            match_state["air_troops_count"] += 1
        else:
            match_state["ground_troops_count"] += 1
    
    # Track estimated total elixir collected
    # Each player gets 1 elixir roughly every 2.8 seconds
    match_state["total_elixir_collected"] = 2 + (my_tower.game_timer / 280)
    
    # Check if opponent might be adapting to our strategy
    if len(match_state["our_tower_health_history"]) >= 4:
        recent_damage = match_state["our_tower_health_history"][-4] - match_state["our_tower_health_history"][-1]
        if recent_damage > 1500:  # Significant recent damage indicates opponent adaptation
            match_state["opponent_adapting"] = True

def update_params_from_signal(signal, current_params):
    """Parse strategy parameters from team_signal if available"""
    params = current_params.copy()
    
    if ":" in signal:  # Only parse if signal contains parameters
        parts = signal.split(",")
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            try:
                # Match signal keys to parameter names
                if key == "e":
                    params["MIN_DEPLOY_ELIXIR"] = float(value)
                elif key == "s":
                    params["SAVE_THRESHOLD"] = float(value)
                elif key == "dt":
                    params["DEFENSIVE_TRIGGER"] = float(value)
                elif key == "y":
                    params["Y_DEFAULT"] = int(value)
                elif key == "xl":
                    params["X_RANGE_LEFT"] = int(value)
                elif key == "xr":
                    params["X_RANGE_RIGHT"] = int(value)
                elif key == "ca":
                    params["AIR_VS_GROUND_BONUS"] = float(value)
                elif key == "cs":
                    params["SPLASH_VS_GROUP_BONUS"] = float(value)
                elif key == "ct":
                    params["TANK_PRIORITY_BONUS"] = float(value)
            except ValueError:
                pass  # Ignore conversion errors
                
    return params

def should_deploy_troops(my_tower, opp_tower, my_troops, opp_troops):
    """Determine if we should deploy troops based on game state"""
    # Get current elixir
    current_elixir = my_tower.total_elixir
    
    # Base elixir threshold
    min_elixir = PARAMS["MIN_DEPLOY_ELIXIR"]
    
    # No available troops to deploy
    if not my_tower.deployable_troops:
        return False
        
    # Early game rush strategy - deploy immediately to establish control
    if match_state["game_phase"] == "early" and my_tower.game_timer < 300:
        min_elixir = min(min_elixir, 3.0)  # Lower threshold early
    
    # Emergency response - deploy when opponent has troops on the field
    if opp_troops and len(opp_troops) > 0:
        # Calculate distance to closest opponent troop
        closest_distance = float('inf')
        for troop in opp_troops:
            # Distance to my tower
            dist = Utils.calculate_distance(troop, (0, 0), False)
            closest_distance = min(closest_distance, dist)
        
        # If opponents are getting close, lower our elixir threshold
        if closest_distance < 30:
            min_elixir = min(min_elixir, PARAMS["EMERGENCY_THRESHOLD"])
        elif closest_distance < 40:
            min_elixir = min(min_elixir, PARAMS["MIN_DEPLOY_ELIXIR"] * 0.85)
    
    # Save elixir for a coordinated push if we're not in danger and have advantage
    if (not opp_troops or len(opp_troops) <= 1) and match_state["game_phase"] != "early":
        my_tower_health_ratio = my_tower.health / 10000  # Assuming max health is 10000
        opp_tower_health_ratio = opp_tower.health / 10000
        
        # We're winning by a good margin, save up for a finishing push
        if my_tower_health_ratio - opp_tower_health_ratio > 0.3:
            if current_elixir < PARAMS["SAVE_THRESHOLD"]:
                return False
    
    # Late game desperation if we're losing badly
    if match_state["game_phase"] == "late":
        my_tower_health_ratio = my_tower.health / 10000
        opp_tower_health_ratio = opp_tower.health / 10000
        
        if my_tower_health_ratio < opp_tower_health_ratio * 0.7:
            # We're losing badly, deploy anything we can
            min_elixir = min(min_elixir, 3.0)
    
    # Check if we meet the elixir threshold
    return current_elixir >= min_elixir

def select_best_troop_and_position(my_tower, opp_tower, my_troops, opp_troops):
    """Select the most effective troop and position based on game state"""
    deployable = my_tower.deployable_troops
    
    if not deployable:
        return None, None
    
    # Calculate our current game state factors
    my_tower_health_ratio = my_tower.health / 10000
    opp_tower_health_ratio = opp_tower.health / 10000
    
    # Determine our mode based on tower health and opponent troops
    in_defensive_mode = my_tower_health_ratio <= PARAMS["DEFENSIVE_TRIGGER"]
    in_aggressive_mode = opp_tower_health_ratio <= PARAMS["AGGRESSION_TRIGGER"]
    
    # Calculate opponent air/ground ratio to inform our troop selection
    air_ground_ratio = 0.5  # Default balanced
    total_troops = match_state["air_troops_count"] + match_state["ground_troops_count"]
    if total_troops > 0:
        air_ground_ratio = match_state["air_troops_count"] / total_troops
    
    # Calculate scores for each deployable troop
    troop_scores = {}
    
    for troop_name in deployable:
        if troop_name not in TROOP_DATA:
            continue
            
        troop_data = TROOP_DATA[troop_name]
        
        # Base score from troop attributes
        base_score = (
            (troop_data["damage"] / 100) +  # Damage contribution
            (troop_data["health"] / 500) +  # Health contribution
            (3 if "high" in troop_data["range"] else
             2 if "medium" in troop_data["range"] else 1)  # Range bonus
        )
        
        # Adjust based on troop type and opponent composition
        type_multiplier = 1.0
        
        # If opponent has mostly ground troops, prioritize air troops
        if air_ground_ratio < 0.3:  # Opponent is ground-heavy
            if troop_data["type"] == "air":
                type_multiplier *= PARAMS["AIR_VS_GROUND_BONUS"]
        
        # If opponent has mostly air troops, prioritize anti-air troops
        elif air_ground_ratio > 0.7:  # Opponent is air-heavy
            if "air" in troop_data["targets"]:
                type_multiplier *= 1.3
        
        # Adjust for multiple opponent troops (splash damage is valuable)
        if opp_troops and len(opp_troops) >= 2:
            if troop_data["splash"]:
                type_multiplier *= PARAMS["SPLASH_VS_GROUP_BONUS"]
        
        # Adjust based on our current mode
        mode_multiplier = 1.0
        
        if in_defensive_mode:
            # In defensive mode, prioritize splash damage and ranged troops
            if troop_data["splash"]:
                mode_multiplier *= 1.4
            if "high" in troop_data["range"]:
                mode_multiplier *= PARAMS["RANGE_BONUS"]
                
        elif in_aggressive_mode:
            # In aggressive mode, prioritize high damage and health
            damage_factor = troop_data["damage"] / 200  # Normalize around ~200 damage
            health_factor = troop_data["health"] / 1000  # Normalize around ~1000 health
            mode_multiplier *= 1.0 + (damage_factor * 0.3) + (health_factor * 0.2)
            
            # Tanks are valuable in aggressive pushes
            if troop_data["health"] > 1500:
                mode_multiplier *= PARAMS["TANK_PRIORITY_BONUS"]
        
        # Calculate final score
        final_score = base_score * type_multiplier * mode_multiplier
        
        # Add the score for this troop
        troop_scores[troop_name] = final_score
    
    # Select the highest scoring troop
    best_troop = max(troop_scores.items(), key=lambda x: x[1])[0] if troop_scores else deployable[0]
    
    # Determine the best position
    position = select_position(best_troop, in_defensive_mode, in_aggressive_mode, my_tower, opp_tower, opp_troops)
    
    return best_troop, position

def select_position(troop_name, defensive_mode, aggressive_mode, my_tower, opp_tower, opp_troops):
    """Select the optimal deployment position based on troop type and game state"""
    # Default position bounds
    x_min = PARAMS["X_RANGE_LEFT"]
    x_max = PARAMS["X_RANGE_RIGHT"]
    
    # Default y position based on mode
    if defensive_mode:
        y_pos = PARAMS["Y_DEFENSIVE"]
    elif aggressive_mode:
        y_pos = PARAMS["Y_AGGRESSIVE"]
    else:
        y_pos = PARAMS["Y_DEFAULT"]
    
    # Adjust x range based on opponent troops
    if opp_troops:
        # Find where opponent troops are concentrated
        opp_x_positions = [troop.position[0] for troop in opp_troops]
        
        if opp_x_positions:
            avg_opp_x = sum(opp_x_positions) / len(opp_x_positions)
            
            if defensive_mode:
                # In defensive mode, deploy to counter opponent troops
                # Deploy closer to opponent's average x position
                x_shift = 10 if avg_opp_x < 0 else -10  # Deploy opposite side of tower
                x_min = max(PARAMS["X_RANGE_LEFT"], avg_opp_x + x_shift - 10)
                x_max = min(PARAMS["X_RANGE_RIGHT"], avg_opp_x + x_shift + 10)
            else:
                # In normal/aggressive mode, avoid opponent concentration
                # Deploy away from where opponent troops are concentrated
                if avg_opp_x < 0:  # Opponents on left side
                    x_min = 0  # Deploy on right side
                    x_max = PARAMS["X_RANGE_RIGHT"]
                else:  # Opponents on right side
                    x_min = PARAMS["X_RANGE_LEFT"]
                    x_max = 0  # Deploy on left side
    
    # Special adjustments for certain troop types
    troop_type = TROOP_DATA.get(troop_name, {}).get("type", "ground")
    troop_range = TROOP_DATA.get(troop_name, {}).get("range", "melee")
    
    # Air troops can be deployed more aggressively
    if troop_type == "air" and not defensive_mode:
        y_pos = min(y_pos + 5, 49)
    
    # Ranged troops should be deployed safer in defensive mode
    if troop_range == "high" and defensive_mode:
        y_pos = max(y_pos - 5, 15)
    
    # Avoid deploying in the exact same spot repeatedly
    if match_state["deploy_positions_used"]:
        # Get the last few deployment positions
        recent_positions = match_state["deploy_positions_used"][-3:]
        
        # Check if we've been deploying in a very small area
        x_coords = [pos[0] for pos in recent_positions]
        if max(x_coords) - min(x_coords) < 10:
            # Force more variation in x position
            if sum(x_coords) / len(x_coords) < 0:  # We've been deploying left
                x_min = 0  # Deploy right instead
            else:  # We've been deploying right
                x_max = 0  # Deploy left instead
    
    # Generate final position
    return (random.randint(x_min, x_max), y_pos)

def deploy_troop(troop, position):
    """Deploy the selected troop at the given position"""
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