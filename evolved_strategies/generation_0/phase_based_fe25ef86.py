from teams.helper_function import Troops, Utils
import random
import math

team_name = "Phase Based 9d41"
troops = [Troops.skeleton, Troops.archer, Troops.knight, Troops.dragon, Troops.musketeer, Troops.valkyrie, Troops.wizard, Troops.giant]
deploy_list = Troops([])
team_signal = ""

# Strategy parameters
params = {
    'early_phase_end': 38,
    'mid_phase_end': 93,
    'early_phase_strategy': {
        'aggression': 0.10707928220761054,
        'elixir_threshold': 8.829709100948525,
        'deploy_distance': 18.15943329968998,
        'troop_preferences': {
            'cheap': 2.0068353168418787,
            'medium': 0.9749573346242998,
            'expensive': 0.20513124447928155,
        },
    },
    'mid_phase_strategy': {
        'aggression': 0.39955416946148864,
        'elixir_threshold': 6.574218877834941,
        'deploy_distance': 9.194770202300173,
        'troop_preferences': {
            'cheap': 0.9755317413044741,
            'medium': 1.8627724122300062,
            'expensive': 1.0847294956563038,
        },
    },
    'late_phase_strategy': {
        'aggression': 0.8293070083737759,
        'elixir_threshold': 4.303943329740864,
        'deploy_distance': 5.6941780241538105,
        'troop_preferences': {
            'cheap': 0.5976253539342049,
            'medium': 1.1249264668197068,
            'expensive': 1.8609332460000143,
        },
    },
    'troop_costs': {
        'Archer': 'cheap',
        'Barbarian': 'cheap',
        'Knight': 'cheap',
        'Minion': 'cheap',
        'Skeleton': 'cheap',
        'Dragon': 'medium',
        'Valkyrie': 'medium',
        'Musketeer': 'medium',
        'Giant': 'expensive',
        'Prince': 'expensive',
        'Wizard': 'expensive',
        'Balloon': 'expensive',
    },
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
    Main logic for the phase-based strategy.
    Adapts strategy based on the current game phase.
    """
    global team_signal
    
    # Access data from the arena
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    troops_data = Troops.troops_data
    
    # Update team signal
    update_team_signal(opp_troops, my_tower)
    
    # Determine current game phase
    current_phase = identify_game_phase(my_tower.game_timer)
    
    # Get phase-specific strategy
    phase_strategy = get_phase_strategy(current_phase)
    
    # Check if we should deploy
    if should_deploy(my_tower.total_elixir, phase_strategy, opp_troops):
        # Select troop based on phase strategy
        troop_to_deploy = select_phase_troop(
            my_tower.deployable_troops,
            my_tower.total_elixir,
            phase_strategy,
            opp_troops
        )
        
        # Deploy if we have a troop selected
        if troop_to_deploy:
            position = determine_position(
                troop_to_deploy,
                phase_strategy,
                my_troops,
                opp_troops,
                troops_data
            )
            deploy_list.list_.append((troop_to_deploy, position))

def update_team_signal(opp_troops, my_tower):
    """
    Update team signal with game state information.
    """
    global team_signal
    
    # Track current game phase
    phase = identify_game_phase(my_tower.game_timer)
    
    # Track our current elixir
    elixir_part = f"phase:{phase},elixir:{int(my_tower.total_elixir)}"
    
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

def identify_game_phase(game_timer):
    """
    Identify current game phase based on game timer.
    Returns: "early", "mid", or "late"
    """
    if game_timer < params["early_phase_end"]:
        return "early"
    elif game_timer < params["mid_phase_end"]:
        return "mid"
    else:
        return "late"

def get_phase_strategy(phase):
    """
    Get strategy parameters for the current phase.
    """
    if phase == "early":
        return params["early_phase_strategy"]
    elif phase == "mid":
        return params["mid_phase_strategy"]
    else:
        return params["late_phase_strategy"]

def should_deploy(current_elixir, phase_strategy, opp_troops):
    """
    Decide whether to deploy based on current phase strategy.
    """
    # Don't deploy if below threshold
    elixir_threshold = phase_strategy["elixir_threshold"]
    if current_elixir < elixir_threshold:
        return False
    
    # Always deploy if at max elixir
    if current_elixir >= 10:
        return True
    
    # More aggressive deployment in late game
    aggression = phase_strategy["aggression"]
    
    # Adjust based on opponent presence
    if opp_troops:
        # More likely to deploy if opponents are present
        aggression *= (1 + 0.1 * len(opp_troops))
    
    # Deploy with probability based on aggression and available elixir
    deploy_probability = aggression * (current_elixir - elixir_threshold) / (10 - elixir_threshold)
    return random.random() < deploy_probability

def select_phase_troop(deployable_troops, available_elixir, phase_strategy, opp_troops):
    """
    Select troop based on phase strategy.
    """
    troops_data = Troops.troops_data
    
    # Find affordable troops
    affordable_troops = []
    for troop in deployable_troops:
        troop_data = troops_data.get(troop, None)
        if troop_data and troop_data.elixir <= available_elixir:
            affordable_troops.append((troop, troop_data))
    
    if not affordable_troops:
        return None
    
    # Calculate scores for each troop
    troop_scores = {}
    
    for troop, troop_data in affordable_troops:
        # Base score based on cost category preference for this phase
        cost_category = params["troop_costs"].get(troop, "medium")
        cost_preference = phase_strategy["troop_preferences"].get(cost_category, 1.0)
        
        # Start with base score from preference
        score = cost_preference
        
        # Adjust based on opponent troops
        if opp_troops:
            # Bonus for troops that can target air if air enemies present
            has_air_enemies = any(t.type == "air" for t in opp_troops)
            if has_air_enemies and troop_data.target_type.get("air", False):
                score *= 1.3
            
            # Bonus for splash damage if many enemies
            if len(opp_troops) >= 3 and troop_data.splash_range > 0:
                score *= 1.2
        
        # Adjust based on aggression level
        aggression = phase_strategy["aggression"]
        
        # Favor direct-attacking troops in aggressive phases
        if aggression > 0.6 and troop_data.attack_range == 0:
            score *= 1.2
        
        # Favor ranged troops in defensive phases
        if aggression < 0.4 and troop_data.attack_range > 0:
            score *= 1.2
        
        troop_scores[troop] = score
    
    # Select troop with highest score
    return max(troop_scores.items(), key=lambda x: x[1])[0]

def determine_position(troop, phase_strategy, my_troops, opp_troops, troops_data):
    """
    Determine optimal position based on phase strategy.
    """
    troop_data = troops_data.get(troop, None)
    
    # Default position
    x = random.uniform(-20, 20)
    y = phase_strategy["deploy_distance"]
    
    # Adjust based on aggression level
    aggression = phase_strategy["aggression"]
    
    if opp_troops:
        # Find center of opponent activity
        avg_x = sum(t.position[0] for t in opp_troops) / len(opp_troops)
        
        # Move toward opponent concentration with randomness
        spread = 15 * (1 - aggression)  # Less spread in aggressive phases
        x = avg_x + random.uniform(-spread, spread)
        
        # Y-position depends on aggression
        # More aggressive = deploy closer to opponent
        defensive_distance = 15 * (1 - aggression)
        y = max(0, defensive_distance)
    else:
        # No opponents, spread across the field
        # More aggressive phases use wider deployment
        spread = 15 * aggression
        x = random.uniform(-spread, spread)
    
    # Check troop type for additional adjustments
    if troop_data and troop_data.type == "air":
        # Air troops can be deployed more aggressively
        y *= 0.7
    
    # Make sure we're in the playable area
    x = max(-25, min(25, x))
    
    return (x, y)