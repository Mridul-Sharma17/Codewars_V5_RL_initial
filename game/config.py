import sys
import os

# Add strategy directories to path
sys.path.append(r"/home/mridul___sharma/Desktop/CodeWars V5/tower_defense_rl/game/teams/baselines")
sys.path.append(r"/home/mridul___sharma/Desktop/CodeWars V5/tower_defense_rl/evolved_strategies/generation_0")

# Import strategy modules
from pratyaksh import *
from counter_picker_41bfbfd0 import *

TEAM1 = pratyaksh
TEAM2 = counter_picker_41bfbfd0
VALUE_ERROR = False
