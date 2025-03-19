from teams.helper_function import Troops, Utils
import random

team_name = "evolved_2"
troops = [
    Troops.dragon, Troops.wizard, 
    Troops.valkyrie, Troops.musketeer,
    Troops.knight, Troops.archer, 
    Troops.minion, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "v:1,l:c,e:5,d:0.9,y:40,t:wmvkd,c:1.3"

def random_x(min_val=-20, max_val=20):
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
    """Parse strategy parameters from team_signal"""
    # Default parameters
    params = {
        "lane": "right",  # Default lane preference
        "min_elixir": 5,  # Minimum elixir to deploy
        "defensive_trigger": 0.9,  # Tower health % to switch to defensive
        "y_default": 40,  # Default y position
        "y_defensive": 20  # Defensive y position
    }
    
    # Parse signal if it contains parameters (v: is the version marker)
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
    """Track opponent troop types in team_signal"""
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
    """Decide whether to deploy troops based on strategy and game state"""
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
    """Choose the best troop to deploy and the position"""
    # Get available troops
    available_troops = my_tower.deployable_troops
    
    if not available_troops:
        return None, None
    
    # Determine whether we're in defensive mode
    defensive_mode = my_tower.health / 10000 < params["defensive_trigger"]
    
    # Choose troop based on strategy
    chosen_troop = None
    
    # Simple heuristic: choose the first available highest-damage troop
    troop_scores = {
        "Wizard": 10 if "Wizard" in available_troops else 0,
        "Dragon": 9 if "Dragon" in available_troops else 0,
        "Musketeer": 8 if "Musketeer" in available_troops else 0,
        "Valkyrie": 7 if "Valkyrie" in available_troops else 0,
        "Knight": 6 if "Knight" in available_troops else 0,
        "Archer": 5 if "Archer" in available_troops else 0,
        "Barbarian": 4 if "Barbarian" in available_troops else 0,
        "Minion": 3 if "Minion" in available_troops else 0
    }
    
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
    
    # Determine position based on lane preference and mode
    x_pos = random_x()
    
    # Adjust position based on lane preference
    if params["lane"] == "left":
        x_pos = random_x(-25, -5)
    elif params["lane"] == "right":
        x_pos = random_x(5, 25)
    elif params["lane"] == "center":
        x_pos = random_x(-10, 10)
    
    # Set y position based on defensive mode
    y_pos = params["y_defensive"] if defensive_mode else params["y_default"]
    
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
