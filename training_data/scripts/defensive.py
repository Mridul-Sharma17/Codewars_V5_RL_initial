from teams.helper_function import Troops, Utils
import random

team_name = "Defensive"
troops = [
    Troops.wizard, Troops.musketeer, 
    Troops.knight, Troops.valkyrie,
    Troops.archer, Troops.dragon, 
    Troops.barbarian, Troops.minion
]
deploy_list = Troops([])
team_signal = "v:1,l:s,e:5.00,s:8.00,dt:0.80,y:30,xl:-15,xr:15,tp:wmkvadbm,ca:1.2,cs:2.0,ct:1.5"

# Strategy parameter constants
MIN_DEPLOY_ELIXIR = 5.0
SAVE_THRESHOLD = 8.0
DEFENSIVE_TRIGGER = 0.8
Y_DEFAULT = 30
Y_DEFENSIVE = 15
X_MIN = -15
X_MAX = 15

def random_x(min_val=X_MIN, max_val=X_MAX):
    return random.randint(min_val, max_val)

def deploy(arena_data: dict):
    """
    DON'T TEMPER DEPLOY FUNCTION
    """
    deploy_list.list_ = []
    logic(arena_data)
    return deploy_list.list_, team_signal

def logic(arena_data: dict):
    global team_signal
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    
    # Track opponent troops in team_signal
    for troop in opp_troops:
        if troop.name not in team_signal:
            if team_signal and not team_signal.endswith(","):
                team_signal += ","
            team_signal += troop.name
    
    # Analyze whether to deploy troops based on elixir and game state
    if should_deploy(my_tower, opp_tower, opp_troops):
        # Choose best troop and position based on strategy
        troop, position = choose_troop_and_position(my_tower, opp_troops)
        if troop:
            deploy_troop(troop, position)

def should_deploy(my_tower, opp_tower, opp_troops):
    """Decide whether to deploy troops based on strategy and game state"""
    # Check if we have enough elixir
    if my_tower.total_elixir < MIN_DEPLOY_ELIXIR:
        return False
        
    # Always deploy if opponent has troops approaching
    if opp_troops and len(opp_troops) > 0:
        return True
        
    # Always deploy early in the game
    if my_tower.game_timer < 300:
        return True
        
    # Deploy based on elixir availability
    if my_tower.total_elixir >= SAVE_THRESHOLD:
        return True
    elif my_tower.total_elixir >= MIN_DEPLOY_ELIXIR + 2:
        return random.random() > 0.3  # 70% chance to deploy
        
    # Default: deploy if we meet minimum elixir
    return my_tower.total_elixir >= MIN_DEPLOY_ELIXIR

def choose_troop_and_position(my_tower, opp_troops):
    """Choose the best troop to deploy and the position"""
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    my_tower_health_ratio = my_tower.health / 10000
    defensive_mode = my_tower_health_ratio < DEFENSIVE_TRIGGER
    
    # Choose troop based on strategy and available troops
    troop_priorities = ['wizard', 'musketeer', 'knight', 'valkyrie', 'archer', 'dragon', 'barbarian', 'minion']
    
    # Score each available troop
    troop_scores = {}
    for troop in available_troops:
        # Higher score = higher priority
        if troop in troop_priorities:
            # Use the index in the priority list (reversed so earlier = higher score)
            troop_scores[troop] = len(troop_priorities) - troop_priorities.index(troop)
        else:
            # Default low score for troops not in priority list
            troop_scores[troop] = 0
    
    # Choose the highest scoring available troop
    chosen_troop = max(troop_scores.items(), key=lambda x: x[1])[0] if troop_scores else available_troops[0]
    
    # Determine position based on lane preference and mode
    lane_preference = "split"
    x_pos = random_x()
    
    # Adjust position based on lane preference
    if lane_preference == "left":
        x_pos = random_x(X_MIN, -5)
    elif lane_preference == "right":
        x_pos = random_x(5, X_MAX)
    elif lane_preference == "center":
        x_pos = random_x(-10, 10)
    elif lane_preference == "split":
        # Alternate between left and right
        if random.random() > 0.5:
            x_pos = random_x(X_MIN, -5)
        else:
            x_pos = random_x(5, X_MAX)
    
    # Set y position based on defensive mode
    y_pos = Y_DEFENSIVE if defensive_mode else Y_DEFAULT
    
    return chosen_troop, (x_pos, y_pos)

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
