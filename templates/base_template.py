from teams.helper_function import Troops, Utils

team_name = "STRATEGY_NAME"
troops = [TROOP_SELECTION]
deploy_list = Troops([])
team_signal = ""

# Strategy parameters
params = PARAMETER_DICT

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
    Main logic for the strategy.
    Will be called by the deploy function.
    """
    global team_signal
    
    # Access data from the arena
    my_tower = arena_data["MyTower"]
    opp_tower = arena_data["OppTower"]
    my_troops = arena_data["MyTroops"]
    opp_troops = arena_data["OppTroops"]
    troops_data = Troops.troops_data
    
    # Check available elixir
    available_elixir = my_tower.total_elixir
    
    # Select best available troop from current cycle
    troop_to_deploy = select_troop(my_tower.deployable_troops, available_elixir, opp_troops)
    
    # Deploy if we have a troop selected and enough elixir
    if troop_to_deploy:
        position = select_position(troop_to_deploy, my_troops, opp_troops, opp_tower)
        deploy_list.list_.append((troop_to_deploy, position))
        
        # Update team signal with information
        update_team_signal(opp_troops)

def select_troop(deployable_troops, available_elixir, opp_troops):
    """
    Select the best troop to deploy based on current conditions.
    """
    # Default simple implementation
    troops_data = Troops.troops_data
    
    # Find troops we can afford
    affordable_troops = []
    for troop in deployable_troops:
        troop_data = troops_data.get(troop, None)
        if troop_data and troop_data.elixir <= available_elixir:
            affordable_troops.append(troop)
    
    # If we can't afford any, return None
    if not affordable_troops:
        return None
    
    # Just return the first affordable troop for now
    return affordable_troops[0]

def select_position(troop, my_troops, opp_troops, opp_tower):
    """
    Select the best position to deploy the troop.
    """
    # Default simple implementation - deploy in the middle
    return (0, 0)  

def update_team_signal(opp_troops):
    """
    Update the team signal based on observation.
    """
    global team_signal
    
    # Example: Track enemy troops
    for troop in opp_troops:
        if troop.name not in team_signal:
            if team_signal:
                team_signal += "," + troop.name
            else:
                team_signal = troop.name