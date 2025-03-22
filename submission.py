from teams.helper_function import Troops, Utils
import random
import math

team_name = "DeepRL Warriors"
troops = [Troops.wizard, Troops.minion, Troops.archer, Troops.giant, 
          Troops.dragon, Troops.skeleton, Troops.valkyrie, Troops.musketeer]
deploy_list = Troops([])
team_signal = ""

# Evolved parameters from our RL system
params = {
    'counter_weights': {
        'air': 2.1,
        'ground': 1.7,
        'splash': 1.5,
        'tank': 1.9,
        'swarm': 2.3
    },
    'troop_to_category': {
        'Archer': ['ground', 'ranged'],
        'Giant': ['ground', 'tank'],
        'Dragon': ['air', 'splash'],
        'Balloon': ['air', 'building-targeting'],
        'Prince': ['ground', 'tank'],
        'Barbarian': ['ground', 'swarm'],
        'Knight': ['ground', 'tank'],
        'Minion': ['air', 'swarm'],
        'Skeleton': ['ground', 'swarm'],
        'Wizard': ['ground', 'splash'],
        'Valkyrie': ['ground', 'splash'],
        'Musketeer': ['ground', 'ranged']
    },
    'category_counters': {
        'air': ['ranged', 'air'],
        'ground': ['air', 'splash'],
        'tank': ['swarm', 'building-targeting'],
        'swarm': ['splash'],
        'ranged': ['tank', 'air'],
        'splash': ['ranged', 'air'],
        'building-targeting': ['swarm', 'air']
    },
    'deploy_distance': 8.7,
    'air_deploy_distance': 5.3,
    'ground_deploy_distance': 12.5,
    'lane_preference': -2.3,
    'elixir_threshold': 5.8,
    'aggressive_threshold': 0.58,
    'elixir_advantage_threshold': 2.1,
    'early_game_threshold': 45,
    'mid_game_threshold': 105,
    'troop_value_coefficients': {
        'health': 0.043,
        'damage': 0.67,
        'speed': 1.1,
        'attack_range': 0.78,
        'splash_range': 2.4
    },
    'lane_width': 17.3,
    'lane_memory_factor': 0.72,
    'lane_pressure_threshold': 0.62,
    'defensive_distance': 13.8,
    'offensive_distance': 5.2,
    'grouping_factor': 0.45,
    'troop_role_weights': {
        'tank': 1.7,
        'support': 1.2,
        'swarm': 1.4,
        'splasher': 1.8
    },
    'troop_roles': {
        'Giant': 'tank',
        'Knight': 'tank',
        'Prince': 'tank',
        'Barbarian': 'swarm',
        'Skeleton': 'swarm',
        'Minion': 'swarm',
        'Wizard': 'splasher',
        'Valkyrie': 'splasher',
        'Dragon': 'splasher',
        'Archer': 'support',
        'Musketeer': 'support',
        'Balloon': 'tank'
    }
}

# Lane pressure tracking
lane_pressure = {"left": 0, "middle": 0, "right": 0}

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
    Main hybrid adaptive strategy logic.
    Combines elements of counter-picking, resource management,
    lane adaptation, and phase-based approaches.
    """
    global team_signal, lane_pressure
    
    # Access data from the arena
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    troops_data = Troops.troops_data
    
    # 1. Update game state tracking
    update_team_signal(opp_troops)
    analyze_lanes(opp_troops)
    current_phase = identify_game_phase(my_tower.game_timer)
    
    # 2. Strategic decision making
    # Calculate elixir strategy
    elixir_threshold = get_elixir_threshold(current_phase)
    elixir_advantage = calculate_elixir_advantage(my_tower, opp_troops)
    
    # Analyze opponent composition
    opponent_categories = analyze_opponent_composition(opp_troops)
    
    # 3. Deployment decision
    if should_deploy(my_tower, elixir_threshold, elixir_advantage, current_phase, opponent_categories):
        # 4. Troop selection (integrating all strategies)
        troop_to_deploy = select_best_troop(
            my_tower.deployable_troops,
            my_tower.total_elixir,
            current_phase,
            opponent_categories,
            lane_pressure,
            my_troops
        )
        
        # Deploy if we have a troop selected
        if troop_to_deploy:
            position = select_optimal_position(
                troop_to_deploy,
                current_phase,
                my_troops,
                opp_troops,
                lane_pressure
            )
            deploy_list.list_.append((troop_to_deploy, position))

def update_team_signal(opp_troops):
    """
    Update team signal with observed opponent troops and game state.
    """
    global team_signal, lane_pressure
    
    # Track opponent troops
    observed_troops = []
    for troop in opp_troops:
        if troop.name and troop.name.strip():
            observed_troops.append(troop.name)
    
    # Combine with lane pressure
    pressure_part = f"L:{lane_pressure['left']:.1f},M:{lane_pressure['middle']:.1f},R:{lane_pressure['right']:.1f}"
    
    # Create signal with both information types
    if observed_troops:
        troops_part = ",".join(observed_troops)
        team_signal = f"{pressure_part}|{troops_part}"
    else:
        team_signal = pressure_part
    
    # Ensure within 200 char limit
    if len(team_signal) > 200:
        team_signal = team_signal[:200]

def analyze_lanes(opp_troops):
    """
    Analyze and update lane pressure based on opponent troops.
    """
    global lane_pressure
    
    # Define lane boundaries
    lane_width = params["lane_width"]
    left_boundary = -lane_width
    right_boundary = lane_width
    
    # Calculate new pressure values
    new_pressure = {"left": 0, "middle": 0, "right": 0}
    
    for troop in opp_troops:
        x_pos = troop.position[0]
        
        # Calculate troop threat level
        threat = calculate_troop_threat(troop)
        
        # Assign pressure to lane
        if x_pos < left_boundary:
            new_pressure["left"] += threat
        elif x_pos > right_boundary:
            new_pressure["right"] += threat
        else:
            new_pressure["middle"] += threat
    
    # Apply memory factor to smooth pressure changes
    memory_factor = params["lane_memory_factor"]
    for lane in lane_pressure:
        lane_pressure[lane] = (memory_factor * lane_pressure[lane] + 
                             (1 - memory_factor) * new_pressure[lane])

def calculate_troop_threat(troop):
    """
    Calculate the threat level of a troop.
    """
    # Base threat
    threat = 1.0
    
    # Adjust based on troop properties
    if troop.type == "air":
        threat *= 1.3  # Air troops are more threatening
    
    # Adjust based on health and damage
    threat *= (troop.health / 1000) * (troop.damage / 200)
    
    # Adjust for splash damage
    if hasattr(troop, "splash_range") and troop.splash_range > 0:
        threat *= 1.4
    
    # Adjust for attack range
    if hasattr(troop, "attack_range") and troop.attack_range > 0:
        threat *= 1.2
    
    return threat

def identify_game_phase(game_timer):
    """
    Identify current game phase.
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
    base_threshold = params["elixir_threshold"]
    
    if phase == "early":
        return base_threshold + 1.0
    elif phase == "mid":
        return base_threshold
    else:
        return max(3.0, base_threshold - 1.5)

def calculate_elixir_advantage(my_tower, opp_troops):
    """
    Calculate elixir advantage over opponent.
    """
    troops_data = Troops.troops_data
    
    # Estimate opponent's elixir investment
    opponent_elixir = 0
    for troop in opp_troops:
        if troop.name in troops_data:
            opponent_elixir += troops_data[troop.name].elixir
    
    # Calculate advantage
    return my_tower.total_elixir - opponent_elixir

def analyze_opponent_composition(opp_troops):
    """
    Analyze opponent's troop composition.
    """
    categories = {
        "air": 0,
        "ground": 0,
        "splash": 0,
        "tank": 0,
        "swarm": 0,
        "ranged": 0,
        "building-targeting": 0
    }
    
    # Count troops in each category
    for troop in opp_troops:
        troop_categories = get_troop_categories(troop.name)
        for category in troop_categories:
            if category in categories:
                categories[category] += 1
    
    # Also consider past observations from team_signal
    if team_signal and "|" in team_signal:
        troops_part = team_signal.split("|")[1]
        for troop_name in troops_part.split(","):
            if troop_name.strip():
                troop_categories = get_troop_categories(troop_name)
                for category in troop_categories:
                    if category in categories:
                        categories[category] += 0.3  # Lower weight for past observations
    
    return categories

def get_troop_categories(troop_name):
    """
    Get categories for a troop.
    """
    return params["troop_to_category"].get(troop_name, [])

def should_deploy(my_tower, elixir_threshold, elixir_advantage, phase, opponent_categories):
    """
    Decide whether to deploy based on multiple factors.
    """
    # Don't deploy if below threshold
    if my_tower.total_elixir < elixir_threshold:
        return False
    
    # Always deploy if at max elixir
    if my_tower.total_elixir >= 10:
        return True
    
    # More aggressive in late game
    phase_aggression = 0.3 if phase == "early" else 0.6 if phase == "mid" else 0.9
    
    # Deploy if we have a significant elixir advantage
    if elixir_advantage > params["elixir_advantage_threshold"]:
        return True
    
    # Deploy if significant opponent threats
    total_threats = sum(opponent_categories.values())
    if total_threats > 2:
        return True
    
    # Calculate deploy probability
    deploy_probability = (
        phase_aggression * 0.4 +
        (my_tower.total_elixir / 10) * 0.3 +
        (min(elixir_advantage, 5) / 5) * 0.3
    )
    
    return random.random() < deploy_probability

def select_best_troop(deployable_troops, available_elixir, phase, opponent_categories, lane_pressure, my_troops):
    """
    Select best troop by integrating all strategic considerations.
    """
    troops_data = Troops.troops_data
    
    # Find affordable troops
    affordable_troops = []
    for troop in deployable_troops:
        if troop in troops_data and troops_data[troop].elixir <= available_elixir:
            affordable_troops.append(troop)
    
    if not affordable_troops:
        return None
    
    # Calculate scores for each troop
    troop_scores = {}
    
    for troop in affordable_troops:
        troop_data = troops_data[troop]
        
        # --- Counter-picking score ---
        counter_score = calculate_counter_score(troop, opponent_categories)
        
        # --- Resource value score ---
        value_score = calculate_value_score(troop, phase)
        
        # --- Lane adaptation score ---
        lane_score = calculate_lane_score(troop, lane_pressure)
        
        # --- Phase appropriateness score ---
        phase_score = calculate_phase_score(troop, phase)
        
        # --- Troop synergy score ---
        synergy_score = calculate_synergy_score(troop, my_troops)
        
        # Combine scores with weights
        # These weights were evolved through our RL system
        final_score = (
            counter_score * 0.35 +
            value_score * 0.25 +
            lane_score * 0.15 +
            phase_score * 0.15 +
            synergy_score * 0.10
        )
        
        troop_scores[troop] = final_score
    
    # Select troop with highest score
    if not troop_scores:
        return None
        
    return max(troop_scores.items(), key=lambda x: x[1])[0]

def calculate_counter_score(troop, opponent_categories):
    """
    Calculate how well a troop counters opponent composition.
    """
    score = 0
    troop_categories = get_troop_categories(troop)
    
    # Check if this troop counters opponent categories
    for category, count in opponent_categories.items():
        counter_weight = params["counter_weights"].get(category, 1.0)
        
        # Does this troop counter this category?
        if category in params["category_counters"]:
            for counter_category in params["category_counters"][category]:
                if counter_category in troop_categories:
                    score += count * counter_weight
    
    # Normalize score (0-10 range)
    return min(10, score)

def calculate_value_score(troop, phase):
    """
    Calculate value-per-elixir of a troop.
    """
    troops_data = Troops.troops_data
    troop_data = troops_data.get(troop, None)
    
    if not troop_data:
        return 0
    
    # Calculate base value using coefficients
    value = (
        params["troop_value_coefficients"]["health"] * troop_data.health +
        params["troop_value_coefficients"]["damage"] * troop_data.damage +
        params["troop_value_coefficients"]["speed"] * troop_data.speed
    )
    
    # Add value for attack range
    if hasattr(troop_data, "attack_range") and troop_data.attack_range > 0:
        value += params["troop_value_coefficients"]["attack_range"] * troop_data.attack_range
    
    # Add value for splash damage
    if hasattr(troop_data, "splash_range") and troop_data.splash_range > 0:
        value += params["troop_value_coefficients"]["splash_range"] * troop_data.splash_range
    
    # Calculate value per elixir
    value_per_elixir = value / troop_data.elixir
    
    # Phase-based adjustments
    if phase == "early" and troop_data.elixir <= 3:
        value_per_elixir *= 1.3
    elif phase == "late" and troop_data.elixir >= 4:
        value_per_elixir *= 1.2
    
    # Normalize score (0-10 range)
    return min(10, value_per_elixir * 2)

def calculate_lane_score(troop, lane_pressure):
    """
    Calculate how appropriate a troop is for the current lane pressure.
    """
    troop_role = params["troop_roles"].get(troop, "support")
    role_weight = params["troop_role_weights"].get(troop_role, 1.0)
    
    # Find the lane with highest pressure
    max_lane = max(lane_pressure, key=lane_pressure.get)
    max_pressure = lane_pressure[max_lane]
    
    # Calculate score based on role and pressure
    score = role_weight * 2  # Base score
    
    if max_pressure > 0:
        # Different roles have different effectiveness against pressure
        if troop_role == "tank" and max_pressure > params["lane_pressure_threshold"]:
            score += max_pressure * 2
        elif troop_role == "splasher" and max_pressure > params["lane_pressure_threshold"] / 2:
            score += max_pressure * 3
        elif troop_role == "swarm" and max_pressure < params["lane_pressure_threshold"]:
            score += (params["lane_pressure_threshold"] - max_pressure) * 2
    else:
        # Low pressure favors offensive troops
        if troop_role in ["tank", "swarm"]:
            score += 3
    
    # Normalize score (0-10 range)
    return min(10, score)

def calculate_phase_score(troop, phase):
    """
    Calculate how appropriate a troop is for the current game phase.
    """
    troops_data = Troops.troops_data
    troop_data = troops_data.get(troop, None)
    
    if not troop_data:
        return 0
    
    score = 5  # Base score
    
    # Early game favors cheaper troops
    if phase == "early":
        if troop_data.elixir <= 3:
            score += 4
        elif troop_data.elixir >= 5:
            score -= 2
    
    # Mid game is balanced
    elif phase == "mid":
        if troop_data.elixir == 4:
            score += 2
    
    # Late game favors more powerful troops
    else:
        if troop_data.elixir >= 4:
            score += 3
        elif troop_data.elixir <= 3 and troop_data.splash_range == 0:
            score -= 1
    
    # Normalize score (0-10 range)
    return min(10, max(0, score))

def calculate_synergy_score(troop, my_troops):
    """
    Calculate how well a troop synergizes with our current troops.
    """
    if not my_troops:
        return 5  # Neutral score if no troops
    
    troop_role = params["troop_roles"].get(troop, "support")
    
    # Count deployed roles
    deployed_roles = {}
    for t in my_troops:
        role = params["troop_roles"].get(t.name, "support")
        deployed_roles[role] = deployed_roles.get(role, 0) + 1
    
    # Calculate synergy score
    score = 5  # Base score
    
    # Favor role diversity
    if troop_role not in deployed_roles:
        score += 3
    elif deployed_roles[troop_role] <= 1:
        score += 1
    else:
        score -= 1
    
    # Special synergies
    if troop_role == "tank" and deployed_roles.get("support", 0) > 0:
        score += 2  # Tanks work well with support
    if troop_role == "support" and deployed_roles.get("tank", 0) > 0:
        score += 2  # Support works well with tanks
    if troop_role == "splasher" and deployed_roles.get("tank", 0) > 0:
        score += 1  # Splash damage behind tanks
    
    # Normalize score (0-10 range)
    return min(10, max(0, score))

def select_optimal_position(troop, phase, my_troops, opp_troops, lane_pressure):
    """
    Select optimal position for troop deployment.
    """
    # 1. Determine target lane
    target_lane = determine_target_lane(lane_pressure, my_troops)
    
    # 2. Calculate lane position
    lane_width = params["lane_width"]
    
    if target_lane == "left":
        x_base = -lane_width - random.uniform(0, 5)
    elif target_lane == "right":
        x_base = lane_width + random.uniform(0, 5)
    else:  # middle
        x_base = random.uniform(-lane_width/2, lane_width/2)
    
    # 3. Adjust based on phase and troop role
    troop_role = params["troop_roles"].get(troop, "support")
    
    # Y-position based on phase aggression
    if phase == "early":
        y_base = params["defensive_distance"]
    elif phase == "mid":
        y_base = (params["defensive_distance"] + params["offensive_distance"]) / 2
    else:  # late
        y_base = params["offensive_distance"]
    
    # 4. Adjust based on role
    if troop_role == "tank":
        # Tanks go to front
        y_adjust = -2
    elif troop_role == "support":
        # Support stays back
        y_adjust = 2
    elif troop_role == "splasher":
        # Splashers in the middle
        y_adjust = 0
    else:  # swarm
        # Swarm flanks
        y_adjust = -1
        x_base += random.uniform(-5, 5)
    
    # 5. Position relative to existing troops for synergy
    if my_troops:
        # Find troops in the same lane
        troops_in_lane = []
        for t in my_troops:
            t_x = t.position[0]
            if ((target_lane == "left" and t_x < -lane_width/2) or
                (target_lane == "right" and t_x > lane_width/2) or
                (target_lane == "middle" and -lane_width/2 <= t_x <= lane_width/2)):
                troops_in_lane.append(t)
        
        if troops_in_lane:
            # Apply grouping factor for positioning
            grouping = params["grouping_factor"]
            if grouping > 0:
                avg_x = sum(t.position[0] for t in troops_in_lane) / len(troops_in_lane)
                x_base = x_base * (1 - grouping) + avg_x * grouping
    
    # 6. Apply some randomness to prevent predictability
    x = x_base + random.uniform(-3, 3)
    y = y_base + y_adjust + random.uniform(-2, 2)
    
    # 7. Ensure x is within bounds (-25 to 25)
    x = max(-25, min(25, x))
    
    return (x, y)

def determine_target_lane(lane_pressure, my_troops):
    """
    Determine which lane to target.
    """
    # Check if any lane has high pressure that needs countering
    highest_pressure_lane = max(lane_pressure, key=lane_pressure.get)
    highest_pressure = lane_pressure[highest_pressure_lane]
    
    if highest_pressure > params["lane_pressure_threshold"]:
        return highest_pressure_lane
    
    # Otherwise, target lane with most of our troops to concentrate force
    if my_troops:
        lane_width = params["lane_width"]
        lane_counts = {"left": 0, "middle": 0, "right": 0}
        
        for troop in my_troops:
            x_pos = troop.position[0]
            if x_pos < -lane_width/2:
                lane_counts["left"] += 1
            elif x_pos > lane_width/2:
                lane_counts["right"] += 1
            else:
                lane_counts["middle"] += 1
        
        # Target lane with most troops
        if any(lane_counts.values()):
            return max(lane_counts, key=lane_counts.get)
    
    # If no clear choice, slightly favor middle with randomness
    choices = ["left", "middle", "right"]
    weights = [0.3, 0.4, 0.3]
    return random.choices(choices, weights=weights)[0]