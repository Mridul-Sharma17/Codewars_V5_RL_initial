# This file tests that pratyaksh.py can be imported
try:
    from pratyaksh import *
    print("pratyaksh module imported successfully!")
    print(f"Team name: {team_name}")
    print(f"Number of troops: {len(troops)}")
except Exception as e:
    print(f"Error importing pratyaksh: {e}")