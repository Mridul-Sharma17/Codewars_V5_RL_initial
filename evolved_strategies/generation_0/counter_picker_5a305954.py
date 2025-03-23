from teams.helper_function import Troops, Utils
import random

team_name = "Counter Picker c705"
troops = [Troops.archer, Troops.dragon, Troops.giant, Troops.barbarian, Troops.minion, Troops.wizard, Troops.prince, Troops.skeleton]
deploy_list = Troops([])
team_signal = ""

# Strategy parameters
params = {
    'counter_weights': {
        'air': 1.2081433137369615,
        'ground': 1.877048529374243,
        'splash': 2.2672196557749533,
        'tank': 0.6737848170085388,
        'swarm': 2.221879735573217,
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
        'Musketeer': ['ground', 'ranged'],
    },
    'category_counters': {
        'air': ['ground', 'ranged'],
        'ground': ['air', 'splash'],
        'tank': ['swarm', 'building-targeting'],
        'swarm': ['splash'],
        'ranged': ['tank'],
        'splash': ['ranged', 'building-targeting'],
        'building-targeting': ['swarm'],
    },
    'deploy_distance': 15.136060309875056,
    'air_deploy_distance': 8.249303303488736,
    'ground_deploy_distance': 13.153144876942484,
    'lane_preference': 18.722539820096465,
    'elixir_threshold': 4.398351364369871,
    'aggressive_threshold': 0.4890624430844651,
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
    Main logic for the counter-picker strategy.
    Analyzes opponent troops and deploys effective counters.
    """
    global team_signal
    
    # Access data from the arena
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    troops_data = Troops.troops_data
    
    # Track opponent troop composition
    update_team_signal(opp_troops)
    
    # Analyze opponent troop categories
    opponent_categories = analyze_opponent_composition(opp_troops)
    
    # Check if we should deploy based on elixir and game state
    if should_deploy(my_tower, opponent_categories):
        # Select best counter troop from current cycle
        troop_to_deploy = select_counter_troop(
            my_tower.deployable_troops, 
            my_tower.total_elixir, 
            opponent_categories
        )
        
        # Deploy if we have a troop selected
        if troop_to_deploy:
            position = select_deploy_position(
                troop_to_deploy, 
                my_troops, 
                opp_troops, 
                opp_tower, 
                troops_data
            )
            deploy_list.list_.append((troop_to_deploy, position))

def update_team_signal(opp_troops):
    """
    Update the team signal with observed opponent troops.
    """
    global team_signal
    
    # Extract current troops from signal
    observed_troops = set(team_signal.split(",")) if team_signal else set()
    
    # Add new troops
    for troop in opp_troops:
        if troop.name and troop.name not in observed_troops and troop.name.strip():
            observed_troops.add(troop.name)
    
    # Update signal (limited to 200 chars)
    team_signal = ",".join(list(observed_troops))
    if len(team_signal) > 200:
        team_signal = team_signal[:200]

def analyze_opponent_composition(opp_troops):
    """
    Analyze opponent's troop composition to determine what categories they're using.
    """
    categories = {
        "air": 0,
        "ground": 0,
        "splash": 0,
        "tank": 0,
        "swarm": 0
    }
    
    # Count troops in each category
    for troop in opp_troops:
        troop_categories = get_troop_categories(troop.name)
        for category in troop_categories:
            if category in categories:
                categories[category] += 1
    
    # Also consider past observations from team_signal
    if team_signal:
        for troop_name in team_signal.split(","):
            if troop_name.strip():
                troop_categories = get_troop_categories(troop_name)
                for category in troop_categories:
                    if category in categories:
                        categories[category] += 0.5  # Lower weight for past observations
    
    return categories

def get_troop_categories(troop_name):
    """
    Get categories for a specific troop.
    """
    # Use predefined categories from parameters
    if troop_name in params["troop_to_category"]:
        return params["troop_to_category"][troop_name]
    return []

def should_deploy(my_tower, opponent_categories):
    """
    Determine if we should deploy a troop based on current conditions.
    """
    # Check if we have enough elixir
    if my_tower.total_elixir < params["elixir_threshold"]:
        return False
    
    # Check if there are opponents worth countering
    total_threats = sum(opponent_categories.values())
    
    # More aggressive deployment if many threats or high elixir
    if total_threats > 3 or my_tower.total_elixir >= 9:
        return True
    
    # Deploy with probability based on elixir amount and threats
    deploy_probability = (my_tower.total_elixir / 10) * (total_threats / 5)
    if deploy_probability > params["aggressive_threshold"]:
        return True
    
    return random.random() < deploy_probability

def select_counter_troop(deployable_troops, available_elixir, opponent_categories):
    """
    Select the best counter troop for the current opponent composition.
    """
    troops_data = Troops.troops_data
    
    # Calculate scores for each deployable troop
    troop_scores = {}
    
    for troop in deployable_troops:
        # Skip if we can't afford this troop
        troop_data = troops_data.get(troop, None)
        if not troop_data or troop_data.elixir > available_elixir:
            continue
            
        # Get this troop's categories
        troop_categories = get_troop_categories(troop)
        
        # Calculate how well this troop counters opponent composition
        score = 0
        
        # Check if this troop counters opponent categories
        for category, count in opponent_categories.items():
            counter_weight = params["counter_weights"].get(category, 1.0)
            
            # Does this troop counter this category?
            if category in params["category_counters"]:
                for counter_category in params["category_counters"][category]:
                    if counter_category in troop_categories:
                        score += count * counter_weight
        
        # If no direct counter, still give some score based on troop strength
        if score == 0 and troop_data:
            # Basic score based on troop stats
            score = 0.1 * (troop_data.damage / 100 + troop_data.health / 1000)
        
        troop_scores[troop] = score
    
    # Select troop with highest score
    if not troop_scores:
        return None
        
    return max(troop_scores.items(), key=lambda x: x[1])[0]

def select_deploy_position(troop, my_troops, opp_troops, opp_tower, troops_data):
    """
    Select the best position to deploy the troop.
    """
    troop_data = troops_data.get(troop, None)
    
    # Default position
    x = random.uniform(-20, 20)
    y = 0
    
    # Adjust based on troop type
    troop_categories = get_troop_categories(troop)
    
    # Check if there are enemy troops to counter
    if opp_troops:
        # Find center of opponent troops
        avg_x = sum(t.position[0] for t in opp_troops) / len(opp_troops)
        
        # Deploy counter troops near opponent troops
        # But not too close to avoid being immediately targeted
        offset = random.uniform(-10, 10)
        x = avg_x + offset
        
        if "air" in troop_categories:
            # Air troops can be deployed more aggressively
            y = params["air_deploy_distance"] 
        else:
            # Ground troops more defensively
            y = params["ground_deploy_distance"]
    else:
        # No opponent troops, deploy towards tower
        # Use lane preference from parameters
        x = params["lane_preference"] + random.uniform(-10, 10)
        
        if "air" in troop_categories:
            y = params["air_deploy_distance"]
        else:
            y = params["ground_deploy_distance"]
    
    # Ensure x is within bounds (-25 to 25)
    x = max(-25, min(25, x))
    
    return (x, y)