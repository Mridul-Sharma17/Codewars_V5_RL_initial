import random

class StrategyGenerator:
    def __init__(self):
        """Initialize the strategy generator"""
        self.troop_stats = {
            "archer": {"elixir": 3, "health": 334, "damage": 118, "type": "ground", "targets": ["air", "ground", "building"], "attack_range": 5},
            "giant": {"elixir": 5, "health": 5423, "damage": 337, "type": "ground", "targets": ["building"], "attack_range": 0},
            "dragon": {"elixir": 4, "health": 1267, "damage": 176, "type": "air", "targets": ["air", "ground", "building"], "attack_range": 3.5},
            "balloon": {"elixir": 5, "health": 2226, "damage": 424, "type": "air", "targets": ["building"], "attack_range": 0},
            "prince": {"elixir": 5, "health": 1920, "damage": 392, "type": "ground", "targets": ["ground", "building"], "attack_range": 0},
            "barbarian": {"elixir": 3, "health": 736, "damage": 161, "type": "ground", "targets": ["ground", "building"], "attack_range": 0},
            "knight": {"elixir": 3, "health": 1938, "damage": 221, "type": "ground", "targets": ["ground", "building"], "attack_range": 0},
            "minion": {"elixir": 3, "health": 252, "damage": 129, "type": "air", "targets": ["air", "ground", "building"], "attack_range": 2},
            "skeleton": {"elixir": 3, "health": 89, "damage": 89, "type": "ground", "targets": ["ground", "building"], "attack_range": 0},
            "wizard": {"elixir": 5, "health": 1100, "damage": 410, "type": "ground", "targets": ["air", "ground", "building"], "attack_range": 5.5},
            "valkyrie": {"elixir": 4, "health": 2097, "damage": 195, "type": "ground", "targets": ["ground", "building"], "attack_range": 0},
            "musketeer": {"elixir": 4, "health": 792, "damage": 239, "type": "ground", "targets": ["air", "ground", "building"], "attack_range": 6}
        }
    
    def generate_code(self, strategy, file_path=None):
        """Generate Python code from strategy parameters"""
        # Create the code template
        code = self._create_code_template(strategy)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(code)
        
        return code
    
    def _create_code_template(self, strategy):
        """Create the actual Python code based on the strategy"""
        # Check if this is the Pratyaksh strategy
        if strategy.get("name") == "pratyaksh" or "pratyaksh" in str(strategy.get("id", "")).lower():
            return self._create_pratyaksh_template()
            
        # Select troops from the strategy
        troop_selection = strategy.get("troop_selection", [
            "dragon", "wizard", "valkyrie", "musketeer",
            "knight", "archer", "minion", "barbarian"
        ])
        
        # Handle troop names properly (capitalize first letter)
        proper_troop_names = [name.capitalize() for name in troop_selection]
        
        # Create signal encoding
        signal = self._encode_strategy(strategy)
        
        # Create the basic template
        code = f'''from teams.helper_function import Troops, Utils
import random

team_name = "{strategy['name']}"
troops = [
    Troops.{proper_troop_names[0].lower()}, Troops.{proper_troop_names[1].lower()}, 
    Troops.{proper_troop_names[2].lower()}, Troops.{proper_troop_names[3].lower()},
    Troops.{proper_troop_names[4].lower()}, Troops.{proper_troop_names[5].lower()}, 
    Troops.{proper_troop_names[6].lower()}, Troops.{proper_troop_names[7].lower()}
]
deploy_list = Troops([])
team_signal = "{signal}"

def random_x(min_val={strategy['position_settings']['x_range'][0]}, max_val={strategy['position_settings']['x_range'][1]}):
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
    params = {{
        "lane": "right",  # Default lane preference
        "min_elixir": {strategy["elixir_thresholds"]["min_deploy"]},  # Minimum elixir to deploy
        "defensive_trigger": {strategy["defensive_trigger"]},  # Tower health % to switch to defensive
        "y_default": {strategy["position_settings"]["y_default"]},  # Default y position
        "y_defensive": {strategy["position_settings"]["defensive_y"]}  # Defensive y position
    }}
    
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
'''
        
        return code
    
    def _create_pratyaksh_template(self):
        """Create the specialized Pratyaksh strategy template"""
        return '''import random
from teams.helper_function import Troops, Utils

team_name = "Pratyaksh"
troops = [
    Troops.wizard, Troops.minion, Troops.archer, Troops.musketeer,
    Troops.dragon, Troops.skeleton, Troops.valkyrie, Troops.barbarian
]
deploy_list = Troops([])
team_signal = "h, Prince, Knight, Barbarian, Princess"

def random_x(min_val=-25, max_val=25):
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
    opp_troops = arena_data["OppTroops"]
    
    # --- Update Team Signal ---
    # Add new opponent troop names (avoid duplicates).
    for troop in opp_troops:
        current_names = [name.strip() for name in team_signal.split(",")] if team_signal else []
        if troop.name not in current_names:
            team_signal = team_signal + ", " + troop.name if team_signal else troop.name
    
    # --- Analyze Opponent's Deck Composition ---
    # Define opponent categories.
    opponent_air = {"Minion", "Dragon", "Musketeer"}
    opponent_ground = {"Prince", "Knight", "Barbarian", "Princess"}
    
    tokens = [token.strip() for token in team_signal.split(",") if token.strip() != "h"]
    count_air = sum(1 for token in tokens if token in opponent_air)
    count_ground = sum(1 for token in tokens if token in opponent_ground)
    
    if count_ground > count_air:
        recommended_counter = "air"    # Counter ground with air units.
    elif count_air > count_ground:
        recommended_counter = "ground" # Counter air with ground units.
    else:
        recommended_counter = None     # No clear preference.
    
    # --- Score Our Troops (only from deployable troops) ---
    deployable = my_tower.deployable_troops
    # Define base scores and categories for our troops.
    troop_data = {
        Troops.wizard:    {"score": 3, "category": "air",    "name": "Wizard"},
        Troops.minion:    {"score": 2, "category": "air",    "name": "Minion"},
        Troops.archer:    {"score": 4, "category": "ground", "name": "Archer"},
        Troops.musketeer: {"score": 3, "category": "ground", "name": "Musketeer"},
        Troops.dragon:    {"score": 5, "category": "air",    "name": "Dragon"},
        Troops.skeleton:  {"score": 2, "category": "ground", "name": "Skeleton"},
        Troops.valkyrie:  {"score": 4, "category": "air",    "name": "Valkyrie"},
        Troops.barbarian: {"score": 3, "category": "ground", "name": "Barbarian"}
    }
    
    bonus = 3  # Bonus for matching the recommended counter strategy.
    best_troop = None
    best_score = -1
    
    # Loop over our full troop list, but only consider those that are deployable.
    for troop in troops:
        if troop not in deployable:
            continue
        base = troop_data[troop]["score"]
        cat = troop_data[troop]["category"]
        score = base + (bonus if recommended_counter and cat == recommended_counter else 0)
        if score > best_score:
            best_score = score
            best_troop = troop

    # --- Deployment Position ---
    if best_troop is not None:
        selected_category = troop_data[best_troop]["category"]
        if selected_category == "air":
            # Deploy air units further forward.
            deploy_position = (random_x(-25, 25), 0)
        else:
            # Deploy ground units slightly closer for support.
            deploy_position = (random_x(-10, 10), 0)
        deploy_list.list_.append((best_troop, deploy_position))
    else:
        # Fallback: If no deployable troop meets criteria, deploy the first available troop.
        if deployable:
            deploy_list.list_.append((deployable[0], (0, 0)))
'''
    
    def _encode_strategy(self, strategy):
        """Encode strategy parameters into a compact team_signal format"""
        parts = [
            f"v:1",  # Version
            f"l:{strategy['lane_preference'][:1]}",  # Lane (first letter)
            f"e:{strategy['elixir_thresholds']['min_deploy']}",  # Min elixir
            f"d:{strategy['defensive_trigger']:.1f}",  # Defensive trigger
            f"y:{strategy['position_settings']['y_default']}"  # Default y position
        ]
        
        # Add troop priority hints if available (first letter of each)
        if "troop_priority" in strategy:
            troop_priority = "".join([t[0] for t in strategy["troop_priority"][:5]])
            parts.append(f"t:{troop_priority}")
        
        # Add counter weights if they differ from default
        if strategy["counter_weights"]["air_vs_ground"] != 1.2:
            parts.append(f"c:{strategy['counter_weights']['air_vs_ground']:.1f}")
            
        # Combine all parts with commas
        return ",".join(parts)