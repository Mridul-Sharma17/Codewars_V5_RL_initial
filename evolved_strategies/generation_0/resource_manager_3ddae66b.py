from teams.helper_function import Troops, Utils
import random

team_name = "Resource Manager ef52"
troops = [Troops.minion, Troops.archer, Troops.knight, Troops.dragon, Troops.musketeer, Troops.valkyrie, Troops.prince, Troops.wizard]
deploy_list = Troops([])
team_signal = ""

# Strategy parameters
params = {
    'early_game_threshold': 31,
    'mid_game_threshold': 96,
    'early_elixir_threshold': 6.695029152805821,
    'mid_elixir_threshold': 7.701616190752526,
    'late_elixir_threshold': 6.34255802679329,
    'elixir_advantage_threshold': 2.0639527858598576,
    'troop_value_coefficients': {
        'health': 0.0991704349875964,
        'damage': 0.12339471218227314,
        'speed': 1.0799423898886196,
        'attack_range': 0.5426179339042219,
        'splash_range': 2.5304874766974725,
    },
    'deploy_distance': 15.086749739096321,
    'spread_factor': 6.027157322731212,
    'lane_preference': -4.748452967345514,
}

def deploy(arena_data: dict):
    """
    Function that returns deployment decisions.
    DO NOT MODIFY THIS FUNCTION.
    """
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    """
    Main logic for the resource manager strategy.
    Focuses on optimal elixir usage throughout different game phases.
    """
    global team_signal
    
    # Access data from the arena
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    troops_data = Troops.troops_data
    
    # Update team signal with game information
    update_team_signal(opp_troops, my_tower)
    
    # Determine current game phase
    current_phase = determine_game_phase(my_tower.game_timer)
    
    # Calculate elixir threshold based on game phase
    elixir_threshold = get_elixir_threshold(current_phase)
    
    # Calculate elixir advantage
    elixir_advantage = calculate_elixir_advantage(my_tower, opp_troops)
    
    # Decide whether to deploy based on elixir management strategy
    if should_deploy(my_tower.total_elixir, elixir_threshold, elixir_advantage, opp_troops):
        # Select optimal troop based on value metrics
        troop_to_deploy = select_value_troop(
            my_tower.deployable_troops, 
            my_tower.total_elixir,
            current_phase,
            opp_troops
        )
        
        # Deploy if we have a troop selected
        if troop_to_deploy:
            position = select_deploy_position(
                troop_to_deploy,
                my_troops,
                opp_troops,
                current_phase,
                troops_data
            )
            deploy_list.list_.append((troop_to_deploy, position))

def update_team_signal(opp_troops, my_tower):
    """
    Update team signal with game state information.
    """
    global team_signal
    
    # Format: "elixir:X,troops:Y,Z,..."
    # Track our current elixir
    elixir_part = f"elixir:{int(my_tower.total_elixir)}"
    
    # Track opponent troops
    troop_names = []
    for troop in opp_troops:
        if troop.name and troop.name.strip():
            troop_names.append(troop.name)
    
    troops_part = f"troops:{','.join(troop_names)}" if troop_names else "troops:none"
    
    # Combine signals
    team_signal = f"{elixir_part},{troops_part}"
    
    # Ensure we don't exceed 200 chars
    if len(team_signal) > 200:
        team_signal = team_signal[:200]

def determine_game_phase(game_timer):
    """
    Determine current game phase based on timer.
    Returns: "early", "mid", or "late"
    """
    if game_timer < params["early_game_threshold"]:
        return "early"
    elif game_timer < params["mid_game_threshold"]:
        return "mid"
    else:
        return "late"

def get_elixir_threshold(phase):
    """
    Get elixir threshold based on game phase.
    """
    if phase == "early":
        return params["early_elixir_threshold"]
    elif phase == "mid":
        return params["mid_elixir_threshold"]
    else:
        return params["late_elixir_threshold"]

def calculate_elixir_advantage(my_tower, opp_troops):
    """
    Calculate our elixir advantage compared to opponent's deployed troops.
    """
    troops_data = Troops.troops_data
    
    # Estimate opponent's elixir investment
    opponent_elixir = 0
    for troop in opp_troops:
        troop_data = troops_data.get(troop.name, None)
        if troop_data:
            opponent_elixir += troop_data.elixir
    
    # Calculate advantage
    return my_tower.total_elixir - opponent_elixir

def should_deploy(current_elixir, threshold, elixir_advantage, opp_troops):
    """
    Decide whether to deploy based on resource management strategy.
    """
    # Don't deploy if below threshold
    if current_elixir < threshold:
        return False
    
    # Always deploy if at max elixir
    if current_elixir >= 10:
        return True
    
    # Deploy if we have a significant elixir advantage
    if elixir_advantage > params["elixir_advantage_threshold"]:
        return True
    
    # Deploy defensively if there are many opponent troops
    if len(opp_troops) >= 3:
        return True
    
    # Otherwise, deploy with probability based on current elixir
    deploy_probability = (current_elixir - threshold) / (10 - threshold)
    return random.random() < deploy_probability

def calculate_troop_value(troop, phase):
    """
    Calculate value-per-elixir of a troop.
    """
    troops_data = Troops.troops_data
    troop_data = troops_data.get(troop, None)
    
    if not troop_data:
        return 0
    
    # Calculate base value using coefficients from parameters
    value = (
        params["troop_value_coefficients"]["health"] * troop_data.health +
        params["troop_value_coefficients"]["damage"] * troop_data.damage +
        params["troop_value_coefficients"]["speed"] * troop_data.speed
    )
    
    # Add value for attack range if applicable
    if hasattr(troop_data, "attack_range") and troop_data.attack_range > 0:
        value += params["troop_value_coefficients"]["attack_range"] * troop_data.attack_range
    
    # Add value for splash damage if applicable
    if hasattr(troop_data, "splash_range") and troop_data.splash_range > 0:
        value += params["troop_value_coefficients"]["splash_range"] * troop_data.splash_range
    
    # Calculate value per elixir
    value_per_elixir = value / troop_data.elixir
    
    # Apply phase-based adjustments
    # In early game, favor cheaper troops
    if phase == "early" and troop_data.elixir <= 3:
        value_per_elixir *= 1.3
    # In late game, favor more powerful troops
    elif phase == "late" and troop_data.elixir >= 4:
        value_per_elixir *= 1.2
    
    return value_per_elixir

def select_value_troop(deployable_troops, available_elixir, phase, opp_troops):
    """
    Select troop with best value-per-elixir.
    """
    troops_data = Troops.troops_data
    
    # Calculate scores for each deployable troop
    troop_scores = {}
    
    for troop in deployable_troops:
        # Skip if we can't afford this troop
        troop_data = troops_data.get(troop, None)
        if not troop_data or troop_data.elixir > available_elixir:
            continue
        
        # Calculate base value score
        value_score = calculate_troop_value(troop, phase)
        
        # Adjust score based on opponent troops
        if opp_troops:
            # If opponent has air troops and this troop can target air, increase score
            has_air_opponents = any(t.type == "air" for t in opp_troops)
            if has_air_opponents and troop_data.target_type.get("air", False):
                value_score *= 1.2
            
            # If opponent has swarm troops and this troop has splash damage, increase score
            swarm_count = sum(1 for t in opp_troops if t.number > 1)
            if swarm_count > 0 and troop_data.splash_range > 0:
                value_score *= (1 + 0.1 * swarm_count)
        
        troop_scores[troop] = value_score
    
    # Select troop with highest score
    if not troop_scores:
        return None
        
    return max(troop_scores.items(), key=lambda x: x[1])[0]

def select_deploy_position(troop, my_troops, opp_troops, phase, troops_data):
    """
    Select optimal position for troop deployment.
    """
    troop_data = troops_data.get(troop, None)
    
    # Default position
    x = random.uniform(-20, 20)
    y = 0
    
    # Adjust based on phase
    if phase == "early":
        # More defensive in early game
        y = params["deploy_distance"] * 0.7
    elif phase == "mid":
        # Balanced in mid game
        y = params["deploy_distance"] * 0.5
    else:
        # More aggressive in late game
        y = params["deploy_distance"] * 0.3
    
    # Adjust position based on opponent troops
    if opp_troops:
        # Find center of opponent troops
        avg_x = sum(t.position[0] for t in opp_troops) / len(opp_troops)
        
        # Deploy where opponent troops are concentrated
        x = avg_x + random.uniform(-params["spread_factor"], params["spread_factor"])
    else:
        # No opponent troops, use lane preference
        x = params["lane_preference"] + random.uniform(-10, 10)
    
    # Ensure x is within bounds (-25 to 25)
    x = max(-25, min(25, x))
    
    return (x, y)