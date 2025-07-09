from teams.helper_function import Troops, Utils
import random
import math

team_name = "Absolute Cinema"
troops = [
    Troops.archer,      # Anti-air, cheap
    Troops.knight,      # Tank, cheap
    Troops.barbarian,   # DPS, cheap
    Troops.musketeer,   # Anti-air, medium
    Troops.dragon,      # Air unit, medium
    Troops.wizard,      # Splash, medium
    Troops.valkyrie,    # Anti-swarm, medium
    Troops.prince       # High damage, expensive
]
deploy_list = Troops([])

# Ultra-compact seed-based representation (65 chars initially)
# s=seed, h=health, e=enemy health, t=turn, a/g=air/ground threat
team_signal = "s:1234,h:100,e:100,t:0,a:0,g:0,p:10,10,10,10,10"

def deploy(arena_data: dict):
    global team_signal
    deploy_list.list_ = []
    
    # Parse state
    state = parse_signal(team_signal)
    
    # Game state analysis
    my_tower = arena_data["MyTower"]
    enemy_tower = arena_data["OppTower"]
    enemy_troops = arena_data["OppTroops"]
    my_troops = arena_data["MyTroops"]
    deployable_troops = my_tower.deployable_troops
    current_elixir = my_tower.total_elixir
    
    # Update turn counter
    turn = state.get("t", 0) + 1
    state["t"] = turn
    
    # Skip if not enough elixir
    if current_elixir < 3:
        team_signal = build_signal(state)
        return deploy_list.list_, team_signal
    
    # Extract critical features
    features = extract_features(arena_data)
    
    # Update state with current features
    state["h"] = int(features["health"] * 100)
    state["e"] = int(features["enemy_health"] * 100)
    state["a"] = int(features["air_threat"] * 100)
    state["g"] = int(features["ground_threat"] * 100)
    
    # Detect health advantage - NEW FEATURE
    health_advantage = features["health"] - features["enemy_health"]
    is_ahead = health_advantage > 0
    
    # ULTRA-DEFENSIVE OVERRIDE: If ahead in health, go full defensive
    if is_ahead:
        defensive_action = ultra_defensive_mode(arena_data, features, state, deployable_troops)
        if defensive_action:
            troop, position = defensive_action
            if troop in deployable_troops:
                deploy_troop(troop, position)
                team_signal = build_signal(state)
                return deploy_list.list_, team_signal
    
    # Critical emergency response - highest priority
    emergency_action = emergency_response(arena_data, features, state)
    if emergency_action:
        troop, position = emergency_action
        if troop in deployable_troops:
            deploy_troop(troop, position)
            team_signal = build_signal(state)
            return deploy_list.list_, team_signal
    
    # Run deep neural network (dynamically generated from seed)
    network = generate_deep_network(state)
    
    # Add health advantage to inputs
    features["health_advantage"] = (health_advantage + 1) / 2  # Normalize to 0-1
    
    # Forward pass with health advantage awareness
    outputs = forward_pass(features, network)
    
    # Modify outputs based on health advantage
    if is_ahead:
        # Bias toward defensive choices when ahead
        outputs[0] = min(0.4, outputs[0])  # Prefer defensive troops
        outputs[2] = max(0.2, min(0.4, outputs[2]))  # Force defensive style
    
    # Convert outputs to action
    action = interpret_outputs(outputs, arena_data, state, deployable_troops, is_ahead)
    
    # Execute action if valid
    if action:
        troop, position = action
        if troop in deployable_troops:
            deploy_troop(troop, position)
            
            # Update neural parameters based on action result
            update_parameters(state, features, action, is_ahead)
    
    # Update signal for next turn
    team_signal = build_signal(state)
    
    return deploy_list.list_, team_signal

def parse_signal(signal):
    """Parse the compact signal into state dictionary"""
    state = {
        "s": 1234,    # Neural seed
        "h": 100,     # Tower health
        "e": 100,     # Enemy tower health
        "t": 0,       # Turn counter
        "a": 0,       # Air threat
        "g": 0,       # Ground threat
        "p": [10, 10, 10, 10, 10]  # Neural meta-parameters
    }
    
    # Parse parts
    parts = signal.split(",")
    for part in parts:
        if ":" in part:
            key, value = part.split(":")
            if key in state:
                if key == "p":
                    # Start meta-parameters array
                    if value:
                        try:
                            state[key][0] = float(value)
                        except (ValueError, IndexError):
                            pass
                else:
                    try:
                        state[key] = float(value)
                    except ValueError:
                        pass
        elif ":" not in part and part:
            # Must be continuing meta-parameters
            param_index = sum(1 for p in parts[:parts.index(part)] if "p:" in p or (p and ":" not in p))
            if param_index < len(state["p"]):
                try:
                    state["p"][param_index] = float(part)
                except ValueError:
                    pass
    
    return state

def build_signal(state):
    """Build compact signal from state"""
    # Convert to integers where possible
    for key in state:
        if key != "p":
            if isinstance(state[key], float) and state[key].is_integer():
                state[key] = int(state[key])
        else:
            for i in range(len(state[key])):
                if isinstance(state[key][i], float) and state[key][i].is_integer():
                    state[key][i] = int(state[key][i])
    
    # Core state - prioritizing most critical elements
    parts = [
        f"s:{state['s']}",  # Neural seed
        f"h:{state['h']}",  # Tower health
        f"e:{state['e']}",  # Enemy health (critical for advantage calculation)
        f"t:{state['t']}",  # Turn counter
        f"a:{state['a']}",  # Air threat
        f"g:{state['g']}",  # Ground threat
    ]
    
    # Neural meta-parameters
    param_str = f"p:{state['p'][0]}"
    for i in range(1, len(state["p"])):
        param_str += f",{state['p'][i]}"
    parts.append(param_str)
    
    # Join all parts
    signal = ",".join(parts)
    
    # Ensure under 200 chars
    if len(signal) >= 195:
        essential = [f"s:{state['s']}", f"h:{state['h']}", f"e:{state['e']}", 
                    f"t:{state['t']}", f"a:{state['a']}", f"g:{state['g']}"]
        signal = ",".join(essential)
    
    return signal

def extract_features(arena_data):
    """Extract comprehensive features for neural processing"""
    my_tower = arena_data["MyTower"]
    enemy_tower = arena_data["OppTower"]
    enemy_troops = arena_data["OppTroops"]
    my_troops = arena_data["MyTroops"]
    
    # Tower health (normalized)
    tower_health = my_tower.health / 1000.0
    
    # Opponent tower health if available
    enemy_health = enemy_tower.health / 1000.0 if hasattr(enemy_tower, "health") else 1.0
    
    # Detailed threat analysis
    air_threat = 0
    ground_threat = 0
    left_threat = 0
    mid_threat = 0
    right_threat = 0
    
    # Distance-based threat levels
    nearest_threat = 100
    critical_zone_threats = 0
    danger_zone_threats = 0
    approach_zone_threats = 0
    
    # Enhanced threat analysis with troop type weighting
    for troop in enemy_troops:
        # Base threat
        threat = 1.0
        
        # Type-specific threat enhancement
        if hasattr(troop, "name"):
            if troop.name in ["Prince", "Dragon", "Wizard"]:
                threat *= 1.5  # Higher threat for powerful units
        
        # Type and position analysis
        is_air = hasattr(troop, "type") and troop.type == "air"
        
        if hasattr(troop, "position") and hasattr(troop.position, "__getitem__"):
            # Distance-based threat assessment
            distance = troop.position[1]
            
            # Track closest enemy
            if distance < nearest_threat:
                nearest_threat = distance
            
            # Zone-based counting with enhanced threat scaling
            if distance < 20:
                # Critical zone (0-20) - extreme threat
                critical_zone_threats += 1
                threat *= 5.0
            elif distance < 35:  # Reduced from 40 to be more sensitive
                # Danger zone (20-35)
                danger_zone_threats += 1
                threat *= 3.0  # Increased from 2.5
            elif distance < 60:
                # Approach zone (40-60)
                approach_zone_threats += 1
                threat *= 1.2
            
            # Lane analysis
            x_pos = troop.position[0]
            if x_pos < -8:
                left_threat += threat
            elif x_pos > 8:
                right_threat += threat
            else:
                mid_threat += threat
        
        # Accumulate threat by type
        if is_air:
            air_threat += threat
        else:
            ground_threat += threat
    
    # Calculate danger proximity (0-1 scale)
    danger_proximity = 0
    if nearest_threat < 100:
        danger_proximity = max(0.0, min(1.0, (80 - nearest_threat) / 80.0))
    
    # Normalize threat values
    max_threat = max(0.1, max(air_threat, ground_threat))
    air_threat = min(1.0, air_threat / max_threat)
    ground_threat = min(1.0, ground_threat / max_threat)
    
    max_lane_threat = max(0.1, max(left_threat, mid_threat, right_threat))
    left_threat = min(1.0, left_threat / max_lane_threat)
    mid_threat = min(1.0, mid_threat / max_lane_threat)
    right_threat = min(1.0, right_threat / max_lane_threat)
    
    # Elixir state
    elixir = my_tower.total_elixir / 10.0  # Normalize to 0-1 range
    
    # Compile feature vector
    features = {
        "health": tower_health,
        "enemy_health": enemy_health,
        "air_threat": air_threat,
        "ground_threat": ground_threat,
        "left_threat": left_threat,
        "mid_threat": mid_threat,
        "right_threat": right_threat,
        "nearest_threat": 1.0 - (nearest_threat / 100.0),  # Invert so closer = higher
        "critical_threats": min(1.0, critical_zone_threats / 3.0),
        "danger_threats": min(1.0, danger_zone_threats / 5.0),
        "approach_threats": min(1.0, approach_zone_threats / 7.0),
        "troop_advantage": min(1.0, (len(my_troops) - len(enemy_troops) + 3) / 6),
        "elixir": elixir,
        "danger_proximity": danger_proximity,
        "turn_progress": min(1.0, my_tower.game_timer / 180.0) if hasattr(my_tower, "game_timer") else 0.5
    }
    
    return features

def ultra_defensive_mode(arena_data, features, state, deployable_troops):
    """Enhanced defensive mode when ahead in health"""
    my_tower = arena_data["MyTower"]
    enemy_troops = arena_data["OppTroops"]
    
    # CRITICAL STRATEGY: Monitor all approaching enemies
    if features["critical_threats"] > 0 or features["danger_threats"] > 0:
        # Find the most dangerous enemy
        most_dangerous = None
        min_distance = 100
        is_dangerous_air = False
        
        for troop in enemy_troops:
            if hasattr(troop, "position") and hasattr(troop.position, "__getitem__"):
                distance = troop.position[1]
                if distance < 35 and distance < min_distance:  # Extended range from 20 to 35
                    min_distance = distance
                    most_dangerous = troop
                    is_dangerous_air = hasattr(troop, "type") and troop.type == "air"
        
        if most_dangerous:
            # Target position slightly ahead of enemy
            x_pos = most_dangerous.position[0]
            y_pos = max(5, most_dangerous.position[1] - 10)
            
            # Select ideal counter based on type
            if is_dangerous_air:
                air_counters = ["Musketeer", "Archer", "Wizard"]
                for counter in air_counters:
                    if counter in deployable_troops:
                        return counter, (x_pos, y_pos)
            else:
                ground_counters = ["Knight", "Valkyrie", "Barbarian"]
                for counter in ground_counters:
                    if counter in deployable_troops:
                        return counter, (x_pos, y_pos)
    
    # PRESSURE RELIEF: Check each lane independently
    lanes = ["left", "mid", "right"]
    lane_pressures = [features["left_threat"], features["mid_threat"], features["right_threat"]]
    max_pressure = max(lane_pressures)
    
    if max_pressure > 0.3:  # Lower threshold for response (was 0.7)
        # Determine which lane needs defense
        target_lane = lanes[lane_pressures.index(max_pressure)]
        
        # Determine x position based on lane
        if target_lane == "left":
            x_pos = -20
        elif target_lane == "right":
            x_pos = 20
        else:
            x_pos = 0
        
        # Defensive y position - further back for maximum protection
        y_pos = 20
        
        # Select counter based on threat composition
        if features["air_threat"] > features["ground_threat"]:
            air_counter = next((t for t in ["Archer", "Musketeer", "Wizard"] if t in deployable_troops), None)
            if air_counter:
                return air_counter, (x_pos, y_pos)
        else:
            ground_counter = next((t for t in ["Knight", "Valkyrie", "Barbarian"] if t in deployable_troops), None)
            if ground_counter:
                return ground_counter, (x_pos, y_pos)
    
    # PREEMPTIVE DEFENSE: Place defenders even with low threat
    if features["air_threat"] > 0.1 or features["ground_threat"] > 0.1:
        # Simple deployments to prepare defenses
        cheap_defenders = ["Knight", "Archer", "Barbarian"]
        for defender in cheap_defenders:
            if defender in deployable_troops:
                # Vary position based on threat distribution
                if features["left_threat"] > features["right_threat"]:
                    return defender, (-15, 25)
                else:
                    return defender, (15, 25)
    
    # MAINTENANCE DEFENSE: Cycle defenders to maintain line
    # Deploy a defensive unit every few turns even without threats
    if state["t"] % 5 == 0:
        # Prefer cheap units for maintenance
        for unit in ["Knight", "Archer", "Barbarian"]:
            if unit in deployable_troops:
                # Alternate lanes
                lane_cycle = state["t"] % 3
                if lane_cycle == 0:
                    return unit, (-20, 30)
                elif lane_cycle == 1:
                    return unit, (0, 25)
                else:
                    return unit, (20, 30)
    
    # No specific defensive action needed
    return None

def emergency_response(arena_data, features, state):
    """Critical emergency response system for immediate threats"""
    enemy_troops = arena_data["OppTroops"]
    
    # CRITICAL THREAT: Enemies extremely close to tower
    if features["critical_threats"] > 0:
        # Find closest critical enemy
        closest_enemy = None
        min_distance = 100
        
        for troop in enemy_troops:
            if hasattr(troop, "position") and hasattr(troop.position, "__getitem__"):
                distance = troop.position[1]
                if distance < 20 and distance < min_distance:
                    min_distance = distance
                    closest_enemy = troop
        
        if closest_enemy:
            # Determine if air or ground
            is_air = hasattr(closest_enemy, "type") and closest_enemy.type == "air"
            
            # Select immediate counter
            counter = select_counter(is_air, "emergency")
            
            # Position for maximum interception
            x_pos = closest_enemy.position[0]
            y_pos = max(5, closest_enemy.position[1] - 10)
            
            return counter, (x_pos, y_pos)
    
    # HIGH DANGER: Multiple enemies in danger zone
    if features["danger_threats"] > 0.4:  # Reduced threshold from 0.6
        # Find center mass of dangerous troops
        danger_troops = []
        for troop in enemy_troops:
            if hasattr(troop, "position") and hasattr(troop.position, "__getitem__"):
                if 20 <= troop.position[1] < 40:
                    danger_troops.append(troop)
        
        if danger_troops:
            # Calculate center of mass
            avg_x = sum(t.position[0] for t in danger_troops) / len(danger_troops)
            
            # Count air vs ground
            air_count = sum(1 for t in danger_troops if hasattr(t, "type") and t.type == "air")
            ground_count = len(danger_troops) - air_count
            
            # Select appropriate counter
            if air_count > ground_count:
                counter = select_counter(True, "high_danger")
            else:
                counter = select_counter(False, "high_danger")
            
            return counter, (avg_x, 25)
    
    # SPECIALIZED THREAT: High air or ground concentration
    if features["air_threat"] > 0.6 or features["ground_threat"] > 0.6:  # Reduced from 0.8
        # Determine which threat dominates
        air_dominated = features["air_threat"] > features["ground_threat"]
        
        # Find center of relevant threats
        threat_center_x = 0
        threat_count = 0
        
        for troop in enemy_troops:
            is_air = hasattr(troop, "type") and troop.type == "air"
            if (air_dominated and is_air) or (not air_dominated and not is_air):
                if hasattr(troop, "position") and hasattr(troop.position, "__getitem__"):
                    threat_center_x += troop.position[0]
                    threat_count += 1
        
        if threat_count > 0:
            threat_center_x /= threat_count
            counter = select_counter(air_dominated, "specialized")
            return counter, (threat_center_x, 30)
    
    # LANE PRESSURE: Heavy pressure in a specific lane
    max_lane_threat = max(features["left_threat"], features["mid_threat"], features["right_threat"])
    if max_lane_threat > 0.6:  # Reduced from 0.7
        # Determine which lane
        if features["left_threat"] == max_lane_threat:
            lane_x = -20
        elif features["right_threat"] == max_lane_threat:
            lane_x = 20
        else:
            lane_x = 0
        
        # Determine if air or ground dominates this lane
        lane_air = 0
        lane_ground = 0
        
        for troop in enemy_troops:
            if hasattr(troop, "position") and hasattr(troop.position, "__getitem__"):
                x_pos = troop.position[0]
                # Check if troop is in the high-pressure lane
                in_lane = False
                if lane_x == -20 and x_pos < -8:
                    in_lane = True
                elif lane_x == 20 and x_pos > 8:
                    in_lane = True
                elif lane_x == 0 and -8 <= x_pos <= 8:
                    in_lane = True
                
                if in_lane:
                    if hasattr(troop, "type") and troop.type == "air":
                        lane_air += 1
                    else:
                        lane_ground += 1
        
        lane_air_dominated = lane_air > lane_ground
        counter = select_counter(lane_air_dominated, "lane_pressure")
        
        return counter, (lane_x, 30)
    
    # No emergency detected
    return None

def select_counter(is_air, situation):
    """Select the optimal counter based on threat type and situation"""
    if is_air:
        if situation == "emergency":
            return "Musketeer"  # Best single-target anti-air
        elif situation == "high_danger":
            return "Wizard"     # Good splash vs air groups
        elif situation == "specialized":
            return "Musketeer"  # Reliable anti-air
        else:  # lane_pressure
            return "Archer"     # Cost-effective anti-air
    else:
        if situation == "emergency":
            return "Knight"     # Tanky, quick deploy
        elif situation == "high_danger":
            return "Valkyrie"   # Excellent vs ground groups
        elif situation == "specialized":
            return "Valkyrie"   # Best ground counter
        else:  # lane_pressure
            return "Knight"     # Reliable ground defender
    
def generate_deep_network(state):
    """Generate a deep neural network from the seed and parameters"""
    seed = int(state["s"])
    params = state["p"]
    
    # Ensure we have enough parameters
    while len(params) < 5:
        params.append(10.0)
    
    # Use seed to generate a random state
    rng = random.Random(seed)
    
    # Define network architecture - SMALLER AND MORE EFFICIENT
    network = {
        "layers": 3,     # 3 hidden layers 
        "width": [8, 6, 4],  # Neurons per layer
        "weights": [],   # Dynamically generated
        "biases": [],    # Dynamically generated
        "activation": "leaky_relu"  # Advanced activation function
    }
    
    # Dynamic weight generation based on seed and parameters
    for l in range(network["layers"] + 1):  # +1 for output layer
        if l == 0:
            # Input to first hidden layer
            input_size = 15  # Feature count
            output_size = network["width"][0]
        elif l == network["layers"]:
            # Last hidden to output
            input_size = network["width"][-1]
            output_size = 3  # 3 outputs
        else:
            # Hidden to hidden
            input_size = network["width"][l-1]
            output_size = network["width"][l]
        
        # Generate weights for this layer
        layer_weights = []
        for i in range(output_size):
            neuron_weights = []
            for j in range(input_size):
                # Get deterministic weight from seed
                weight_seed = (seed + l*10000 + i*100 + j) % 10000
                base_weight = (weight_seed % 100) / 50.0 - 1.0  # Range -1 to 1
                
                # Apply parameter scaling
                param_idx = (i + j) % len(params)
                param_effect = params[param_idx] / 10.0
                
                # Special emphasis on defensive parameters when l=0 (input layer)
                if l == 0 and j < 10:  # First 10 features are threat-related
                    param_effect *= 1.2  # 20% boost to defensive sensitivity
                
                final_weight = base_weight * param_effect
                neuron_weights.append(final_weight)
            layer_weights.append(neuron_weights)
        
        network["weights"].append(layer_weights)
    
    # Generate biases similarly
    for l in range(network["layers"] + 1):
        if l == network["layers"]:
            # Output layer
            size = 3
        else:
            # Hidden layer
            size = network["width"][l]
        
        layer_biases = []
        for i in range(size):
            bias_seed = (seed + 20000 + l*1000 + i) % 10000
            base_bias = (bias_seed % 100) / 100.0 - 0.5  # Range -0.5 to 0.5
            
            param_idx = (l + i) % len(params)
            param_effect = params[param_idx] / 20.0
            
            final_bias = base_bias * param_effect
            layer_biases.append(final_bias)
        
        network["biases"].append(layer_biases)
    
    return network

def forward_pass(features, network):
    """Run forward pass through the deep neural network"""
    # Create input vector (ordered list of feature values)
    inputs = [
        features["health"],
        features["enemy_health"],
        features["air_threat"],
        features["ground_threat"],
        features["left_threat"],
        features["mid_threat"],
        features["right_threat"],
        features["nearest_threat"],
        features["critical_threats"],
        features["danger_threats"],
        features["approach_threats"],
        features["troop_advantage"],
        features["elixir"],
        features["danger_proximity"],
        features["health_advantage"]  # Our new health advantage feature
    ]
    
    # Forward pass through hidden layers
    activations = inputs
    
    for l in range(network["layers"] + 1):  # +1 for output layer
        if l < len(network["weights"]) and l < len(network["biases"]):
            layer_weights = network["weights"][l]
            layer_biases = network["biases"][l]
            
            # Compute next layer activations
            next_activations = []
            
            for i in range(len(layer_weights)):
                neuron_weights = layer_weights[i]
                neuron_bias = layer_biases[i]
                
                # Weighted sum
                z = neuron_bias
                for j in range(len(neuron_weights)):
                    if j < len(activations):
                        z += neuron_weights[j] * activations[j]
                
                # Apply activation function (last layer = sigmoid, others = leaky ReLU)
                if l == network["layers"]:
                    # Sigmoid for output layer
                    a = 1.0 / (1.0 + math.exp(-z))
                else:
                    # Leaky ReLU for hidden layers
                    a = z if z > 0 else 0.01 * z
                
                next_activations.append(a)
            
            activations = next_activations
    
    # Health advantage directly influences the output
    # The higher our health advantage, the more defensive we become
    if len(activations) >= 3 and features["health_advantage"] > 0.5:
        # Scale down aggressiveness when ahead
        advantage_factor = features["health_advantage"] - 0.5  # 0-0.5 range
        activations[0] = max(0.1, activations[0] * (1 - advantage_factor))  # More defensive troops
        activations[2] = max(0.1, activations[2] * (1 - advantage_factor * 2))  # Much more defensive style
    
    return activations

def interpret_outputs(outputs, arena_data, state, deployable_troops, is_ahead):
    """Convert neural outputs to concrete actions with health advantage awareness"""
    my_tower = arena_data["MyTower"]
    current_elixir = my_tower.total_elixir
    
    # Output interpretation
    troop_value = outputs[0]    # What type of troop to deploy (0-1)
    lane_value = outputs[1]     # Which lane to deploy in (0-1)
    style_value = outputs[2]    # How aggressive to be (0-1)
    
    # HEALTH ADVANTAGE OVERRIDE - force defensive when ahead
    if is_ahead:
        # Force defensive style
        style_value *= 0.5  # Cut aggressiveness in half
        
        # Cap troop value to prefer defensive units
        troop_value = min(0.4, troop_value)
    
    # Adjust style based on tower health
    health_percent = state["h"] / 100.0
    if health_percent < 0.6:
        # Low health forces defensive style
        style_value *= health_percent  # Scale down aggressiveness
    
    # Troop selection based on neural output and game state
    troop = select_troop(troop_value, style_value, arena_data, state, deployable_troops, current_elixir, is_ahead)
    
    # Lane selection
    lane = select_lane(lane_value, state)
    
    # Position calculation - more defensive when ahead
    position = calculate_position(lane, style_value, state, is_ahead)
    
    return troop, position

def select_troop(troop_value, style_value, arena_data, state, deployable_troops, elixir, is_ahead):
    """Select optimal troop based on neural preference and game state"""
    # Group troops by cost and role
    defensive_cheap = ["Knight", "Archer", "Barbarian"]
    defensive_medium = ["Musketeer", "Valkyrie"]
    offensive_medium = ["Dragon", "Wizard"]
    offensive_expensive = ["Prince"]
    
    # Check for specific threats that need counters
    if state["a"] > 20:  # More sensitive air threat detection (was 30)
        air_counters = ["Musketeer", "Archer", "Wizard", "Dragon"]
        for counter in air_counters:
            if counter in deployable_troops:
                return counter
    
    if state["g"] > 20:  # More sensitive ground threat detection (was 30)
        ground_counters = ["Valkyrie", "Knight", "Wizard"]
        for counter in ground_counters:
            if counter in deployable_troops:
                return counter
    
    # When ahead in health, strongly prefer defensive troops
    if is_ahead:
        # Mostly cheap defensive options when ahead
        if troop_value < 0.7 or elixir < 4:
            candidates = defensive_cheap
        else:
            candidates = defensive_medium
    # Normal classification otherwise
    elif style_value < 0.3:  # Defensive
        if troop_value < 0.5 or elixir < 4:
            candidates = defensive_cheap
        else:
            candidates = defensive_medium
    elif style_value < 0.7:  # Balanced
        if troop_value < 0.3:
            candidates = defensive_cheap
        elif troop_value < 0.7 or elixir < 5:
            candidates = defensive_medium + offensive_medium
        else:
            candidates = offensive_medium
    else:  # Aggressive (only when not ahead)
        if troop_value < 0.3 or elixir < 4:
            candidates = defensive_cheap
        elif troop_value < 0.6 or elixir < 5:
            candidates = offensive_medium
        else:
            candidates = offensive_expensive + offensive_medium
    
    # Filter by what's actually deployable
    available = [t for t in candidates if t in deployable_troops]
    
    # Fallback to any deployable
    if not available:
        available = [t for t in deployable_troops]
    
    if not available:
        return None
    
    # Return highest priority troop
    return available[0]

def select_lane(lane_value, state):
    """Determine optimal lane based on neural preference and threats"""
    # Convert neural output to lane
    if lane_value < 0.33:
        neural_lane = "left"
    elif lane_value < 0.66:
        neural_lane = "mid"
    else:
        neural_lane = "right"
    
    # Check for lane overrides based on threats
    left_threat = state.get("l", 0)
    right_threat = state.get("r", 0)
    
    # If one lane has more pressure, defend it
    if left_threat > right_threat * 1.2:  # More sensitive (was 1.5)
        return "left"
    elif right_threat > left_threat * 1.2:  # More sensitive (was 1.5)
        return "right"
    
    # Otherwise use neural preference
    return neural_lane

def calculate_position(lane, style_value, state, is_ahead):
    """Calculate optimal deployment position with health advantage awareness"""
    # Base Y position (0-49, higher = further from tower)
    if is_ahead:
        # When ahead in health, deploy more defensively
        if style_value < 0.3:  # Very defensive
            y_pos = 15  # Very close to tower
        elif style_value < 0.6:  # Defensive
            y_pos = 20  # Close to tower
        else:  # Balanced at most
            y_pos = 30  # Middle position
    else:
        # Normal positioning when not ahead
        if style_value < 0.2:  # Very defensive
            y_pos = 15
        elif style_value < 0.4:  # Defensive
            y_pos = 25
        elif style_value < 0.6:  # Balanced
            y_pos = 35
        elif style_value < 0.8:  # Aggressive
            y_pos = 42
        else:  # Very aggressive
            y_pos = 49
    
    # X position based on lane with slight randomization
    if lane == "left":
        x_pos = random.uniform(-25, -15)
    elif lane == "right":
        x_pos = random.uniform(15, 25)
    else:  # mid
        x_pos = random.uniform(-7, 7)
    
    return (x_pos, y_pos)

def update_parameters(state, features, action, is_ahead):
    """Update neural parameters based on current game state"""
    # Extract current meta-parameters
    params = state["p"]
    
    # Ensure we have enough parameters
    while len(params) < 5:
        params.append(10.0)
    
    # Health-advantage adaptive learning
    if is_ahead:
        # When ahead, strengthen defensive parameters
        params[0] += 0.3  # First parameter affects troop selection
        params[2] -= 0.2  # Third parameter affects style
    
    # Adaptation depends on game state
    health = features["health"]
    
    # Calculate adaptation direction
    # If health is decreasing, strengthen defensive parameters
    if health < state["h"]/100.0:
        # Health decreased - emergency defensive shift
        params[0] += 0.8  # Boost parameter that affects defensive neurons
        params[2] -= 0.5  # Reduce parameter that affects offensive neurons
    
    # If under active threat, boost response parameters
    if features["critical_threats"] > 0 or features["danger_threats"] > 0.5:
        params[1] += 0.6  # Second parameter affects threat recognition
        params[3] += 0.4  # Fourth parameter affects lane selection
    
    # If we have a large health advantage, lock in defensive strategy
    health_advantage = features["health"] - features["enemy_health"]
    if health_advantage > 0.3:  # 30% health advantage
        # Strong commitment to defense
        params[0] = max(params[0], 15.0)  # Ensure defensive parameter is high
        params[2] = min(params[2], 8.0)   # Ensure offensive parameter is low
    
    # Adaptive learning rate - decreases over time for stability
    turn_factor = 1.0 / (1.0 + state["t"] / 50.0)
    
    # Apply random exploration only when not ahead in health
    if not is_ahead:
        random_idx = int(state["t"]) % len(params)
        exploration = (random.random() - 0.5) * turn_factor * 2
        params[random_idx] += exploration
    
    # Ensure parameters stay within reasonable bounds
    for i in range(len(params)):
        params[i] = max(1.0, min(20.0, params[i]))
    
    # Update state with new parameters
    state["p"] = params

def deploy_troop(troop_type, position):
    """Deploy a specific troop at the given position"""
    if troop_type == "Archer":
        deploy_list.deploy_archer(position)
    elif troop_type == "Knight":
        deploy_list.deploy_knight(position)
    elif troop_type == "Prince":
        deploy_list.deploy_prince(position)
    elif troop_type == "Musketeer":
        deploy_list.deploy_musketeer(position)
    elif troop_type == "Dragon":
        deploy_list.deploy_dragon(position)
    elif troop_type == "Barbarian":
        deploy_list.deploy_barbarian(position)
    elif troop_type == "Valkyrie":
        deploy_list.deploy_valkyrie(position)
    elif troop_type == "Wizard":
        deploy_list.deploy_wizard(position)