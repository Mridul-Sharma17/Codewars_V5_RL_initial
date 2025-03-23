from teams.helper_function import Troops, Utils
import random
import math

team_name = "Lane Adaptor 226d"
troops = [Troops.prince, Troops.knight, Troops.musketeer, Troops.archer, Troops.valkyrie, Troops.dragon, Troops.barbarian, Troops.minion]
deploy_list = Troops([])
team_signal = ""

# Strategy parameters
params = {
    'lane_width': 23.777415210587,
    'lane_memory_factor': 0.6750265553674138,
    'lane_pressure_threshold': 0.5013751986388493,
    'defensive_distance': 11.481366498274115,
    'offensive_distance': 4.314175814376984,
    'grouping_factor': 0.9141605770628177,
    'counter_deploy_ratio': 1.0379865481627801,
    'troop_role_weights': {
        'tank': 1.920621620666326,
        'support': 1.5623300229084465,
        'swarm': 0.9574576843574073,
        'splasher': 1.4665429435274593,
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
        'Balloon': 'tank',
    },
}

# Memory for tracking lane pressure
lane_pressure = {"left": 0, "middle": 0, "right": 0}
enemy_deployment_history = []

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
    Main logic for the lane adaptor strategy.
    Analyzes lane pressure and adapts deployments accordingly.
    """
    global team_signal, lane_pressure, enemy_deployment_history
    
    # Access data from the arena
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    troops_data = Troops.troops_data
    
    # Update team signal
    update_team_signal(opp_troops)
    
    # Analyze lane pressure
    analyze_lanes(opp_troops)
    
    # Update enemy deployment history
    update_enemy_history(opp_troops)
    
    # Check if we should deploy
    if should_deploy(my_tower, lane_pressure):
        # Select lane-appropriate troop
        troop_to_deploy, target_lane = select_lane_troop(
            my_tower.deployable_troops,
            my_tower.total_elixir,
            lane_pressure,
            my_troops
        )
        
        # Deploy if we have a troop selected
        if troop_to_deploy:
            position = calculate_lane_position(
                troop_to_deploy,
                target_lane,
                opp_troops,
                troops_data
            )
            deploy_list.list_.append((troop_to_deploy, position))

def update_team_signal(opp_troops):
    """
    Update team signal with lane pressure information.
    """
    global team_signal, lane_pressure
    
    # Format: "L:X,M:Y,R:Z" where X,Y,Z are pressure values
    pressure_signal = f"L:{lane_pressure['left']:.1f},M:{lane_pressure['middle']:.1f},R:{lane_pressure['right']:.1f}"
    
    # Add opponent troop information if space allows
    troop_counts = {}
    for troop in opp_troops:
        if troop.name not in troop_counts:
            troop_counts[troop.name] = 0
        troop_counts[troop.name] += 1
    
    troop_signal = ",".join([f"{name}:{count}" for name, count in troop_counts.items()])
    
    # Combine signals
    team_signal = pressure_signal
    if troop_signal and len(pressure_signal) + len(troop_signal) + 1 <= 200:
        team_signal += "," + troop_signal
    
    # Ensure we don't exceed 200 chars
    if len(team_signal) > 200:
        team_signal = team_signal[:200]

def analyze_lanes(opp_troops):
    """
    Analyze lane pressure from opponent troops.
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
        pressure_value = 1.0
        
        # Adjust pressure based on troop type
        if troop.type == "air":
            pressure_value *= 1.5  # Air troops are more threatening
        
        # Adjust pressure based on troop health
        pressure_value *= troop.health / 1000
        
        # Assign pressure to lane
        if x_pos < left_boundary:
            new_pressure["left"] += pressure_value
        elif x_pos > right_boundary:
            new_pressure["right"] += pressure_value
        else:
            new_pressure["middle"] += pressure_value
    
    # Apply memory factor to smooth pressure changes
    memory_factor = params["lane_memory_factor"]
    for lane in lane_pressure:
        lane_pressure[lane] = (memory_factor * lane_pressure[lane] + 
                              (1 - memory_factor) * new_pressure[lane])

def update_enemy_history(opp_troops):
    """
    Update history of enemy deployments.
    """
    global enemy_deployment_history
    
    # Add new troops to history
    for troop in opp_troops:
        if not any(t["uid"] == troop.uid for t in enemy_deployment_history):
            enemy_deployment_history.append({
                "uid": troop.uid,
                "name": troop.name,
                "position": troop.position,
                "type": troop.type
            })
    
    # Limit history size
    if len(enemy_deployment_history) > 20:
        enemy_deployment_history = enemy_deployment_history[-20:]

def should_deploy(my_tower, lane_pressure):
    """
    Decide whether to deploy based on lane pressure and elixir.
    """
    # Check if we have enough elixir
    min_elixir = 3  # Minimum elixir needed for any troop
    if my_tower.total_elixir < min_elixir:
        return False
    
    # Always deploy if at max elixir
    if my_tower.total_elixir >= 10:
        return True
    
    # Calculate highest lane pressure
    max_pressure = max(lane_pressure.values())
    
    # Deploy if pressure exceeds threshold
    if max_pressure > params["lane_pressure_threshold"]:
        return True
    
    # Otherwise, deploy with probability based on elixir amount
    deploy_probability = my_tower.total_elixir / 10
    return random.random() < deploy_probability

def select_lane_troop(deployable_troops, available_elixir, lane_pressure, my_troops):
    """
    Select appropriate troop for current lane pressure.
    Returns (troop, target_lane)
    """
    troops_data = Troops.troops_data
    
    # Find affordable troops
    affordable_troops = []
    for troop in deployable_troops:
        troop_data = troops_data.get(troop, None)
        if troop_data and troop_data.elixir <= available_elixir:
            affordable_troops.append(troop)
    
    if not affordable_troops:
        return None, None
    
    # Identify the lane with highest pressure to counter
    target_lane = max(lane_pressure, key=lane_pressure.get)
    
    # If no significant pressure, pick lane with most of our troops
    if lane_pressure[target_lane] < params["lane_pressure_threshold"] / 2:
        lane_counts = {"left": 0, "middle": 0, "right": 0}
        lane_width = params["lane_width"]
        
        for troop in my_troops:
            x_pos = troop.position[0]
            if x_pos < -lane_width:
                lane_counts["left"] += 1
            elif x_pos > lane_width:
                lane_counts["right"] += 1
            else:
                lane_counts["middle"] += 1
        
        # Target lane with most of our troops to concentrate force
        if any(lane_counts.values()):
            target_lane = max(lane_counts, key=lane_counts.get)
    
    # Calculate scores for each affordable troop
    troop_scores = {}
    
    for troop in affordable_troops:
        # Get troop role
        troop_role = params["troop_roles"].get(troop, "support")
        role_weight = params["troop_role_weights"].get(troop_role, 1.0)
        
        # Calculate score based on lane pressure and role
        score = role_weight
        
        # Adjust score based on counter relationships
        if lane_pressure[target_lane] > 0:
            # If lane has high pressure, prefer tanks and splash damage
            if troop_role == "tank":
                score *= (1 + lane_pressure[target_lane])
            elif troop_role == "splasher":
                score *= (1 + 0.5 * lane_pressure[target_lane])
        else:
            # If lane has low pressure, prefer swarm and support troops
            if troop_role == "swarm" or troop_role == "support":
                score *= 1.5
        
        # Adjust based on currently deployed troops
        deployed_roles = [params["troop_roles"].get(t.name, "support") for t in my_troops]
        if troop_role not in deployed_roles and deployed_roles:
            # Bonus for diversifying roles
            score *= 1.2
        
        troop_scores[troop] = score
    
    # Select troop with highest score
    if not troop_scores:
        return None, None
        
    best_troop = max(troop_scores.items(), key=lambda x: x[1])[0]
    return best_troop, target_lane

def calculate_lane_position(troop, target_lane, opp_troops, troops_data):
    """
    Calculate optimal position in the target lane.
    """
    lane_width = params["lane_width"]
    troop_data = troops_data.get(troop, None)
    
    # Default position
    x = 0
    y = 0
    
    # Set x-coordinate based on target lane
    if target_lane == "left":
        x = -lane_width - random.uniform(0, 5)
    elif target_lane == "right":
        x = lane_width + random.uniform(0, 5)
    else:  # middle
        x = random.uniform(-lane_width/2, lane_width/2)
    
    # Set y-coordinate based on troop role and opponent presence
    troop_role = params["troop_roles"].get(troop, "support")
    
    # Count opponent troops in the target lane
    lane_opponents = []
    for opp in opp_troops:
        opp_x = opp.position[0]
        if ((target_lane == "left" and opp_x < -lane_width) or
            (target_lane == "right" and opp_x > lane_width) or
            (target_lane == "middle" and -lane_width <= opp_x <= lane_width)):
            lane_opponents.append(opp)
    
    if lane_opponents:
        # Defensive positioning if opponents in lane
        if troop_role == "tank":
            # Tanks go to front
            y = params["defensive_distance"] * 0.8
        elif troop_role == "support":
            # Support stays back
            y = params["defensive_distance"] * 1.2
        elif troop_role == "splasher":
            # Splashers in the middle
            y = params["defensive_distance"]
        else:  # swarm
            # Swarm troops can be more aggressive
            y = params["defensive_distance"] * 0.7
    else:
        # Offensive positioning if no opponents
        if troop_role == "tank":
            # Tanks lead the charge
            y = params["offensive_distance"] * 0.5
        elif troop_role == "swarm":
            # Swarm follows closely
            y = params["offensive_distance"] * 0.7
        else:
            # Others follow
            y = params["offensive_distance"] * 1.0
    
    # Add grouping factor to concentrate troops
    grouping = params["grouping_factor"]
    if grouping > 0:
        # Find any of our troops in this lane
        lane_width = params["lane_width"]
        our_troops_in_lane = []
        
        if "my_troops" in locals():
            for t in my_troops:
                t_x = t.position[0]
                if ((target_lane == "left" and t_x < -lane_width) or
                    (target_lane == "right" and t_x > lane_width) or
                    (target_lane == "middle" and -lane_width <= t_x <= lane_width)):
                    our_troops_in_lane.append(t)
        
        # Position closer to our troops
        if our_troops_in_lane:
            avg_x = sum(t.position[0] for t in our_troops_in_lane) / len(our_troops_in_lane)
            avg_y = sum(t.position[1] for t in our_troops_in_lane) / len(our_troops_in_lane)
            
            # Move toward average position, weighted by grouping factor
            x = x * (1 - grouping) + avg_x * grouping
            y = y * (1 - grouping) + avg_y * grouping
    
    # Ensure x is within bounds (-25 to 25)
    x = max(-25, min(25, x))
    
    return (x, y)