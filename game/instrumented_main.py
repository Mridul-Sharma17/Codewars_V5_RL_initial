import os
import json
import sys
import time

from scripts.game_config import *

def extract_game_results(game):
    """Extract results from the game object"""
    results = {
        "winner": game.winner,
        "team1_name": game.team_name1,
        "team2_name": game.team_name2,
        "team1_health": game.tower1.health if hasattr(game, "tower1") else 0,
        "team2_health": game.tower2.health if hasattr(game, "tower2") else 0,
        "game_duration": game.game_counter,
        "message": game.message
    }
    return results

from game import Game
import inspect
from config import TEAM1, TEAM2

def validate_module(module, name):
    attributes = dir(module)
    
    # Expected variables and classes
    expected_variables = {"team_name", "troops", "deploy_list", "team_signal"}
    expected_classes = {"Troops", "Utils"}
    
    # Extract variables (excluding functions, classes, and modules)
    variables = {
        attr for attr in attributes
        if not callable(getattr(module, attr))
        and not attr.startswith("__")
        and not inspect.ismodule(getattr(module, attr))
        and not inspect.isclass(getattr(module, attr))
    }
    
    # Extract classes
    classes = {
        attr for attr in attributes
        if inspect.isclass(getattr(module, attr))
    }
    
    # Condition 1: Check for exact variables and classes
    if variables != expected_variables:
        print(f"Fail: Variables do not match. Found: {variables} for {name}")
        return False
    
    if classes != expected_classes:
        print(f"Fail: Classes do not match. Found: {classes} for {name}")
        return False
    
    # Condition 3: Check len(set(troops)) == 8
    if len(set(module.troops)) != 8 or len(module.troops) != 8:
        print(f"Fail: troops does not contain exactly 8 unique elements for {name}")
        return False
    
    print(f"Pass: All conditions met for {name} : {module.team_name}!")

    return True

team1_test_pass = False
team2_test_pass = False

team1_test_pass = validate_module(TEAM1, "TEAM 1") or team1_test_pass
team2_test_pass = validate_module(TEAM2, "TEAM 2") or team2_test_pass


if team1_test_pass and team2_test_pass:
    # Create game instance
    game = Game(TEAM1.troops,TEAM2.troops,TEAM1.team_name,TEAM2.team_name)
    
    # Check for headless mode
    if 'HEADLESS_MODE' in globals() and HEADLESS_MODE:
        # Run the game in pure headless mode without any pygame display components
        try:
            import pygame
            pygame.init()
            headless_simulation(game)
        except Exception as e:
            print(f"Error in headless simulation: {e}")
            sys.exit(1)
    else:
        # Regular display mode - override run method to auto-close when finished
        original_run = game.run
        def modified_run():
            # IMPORTANT: Explicitly set fps to 500 (fast speed)
            game.fps = 500
            
            # Set up variables to track game state
            game_end_time = 1830  # Default
            if 'GAME_END_TIME' in globals():
                game_end_time = GAME_END_TIME
                
            # Auto-close detection
            running = True
            
            # Run the game loop with high speed settings
            while running:
                # Render the game
                game.render_game_screen()
                game.render_left_screen()
                game.render_right_screen()
                
                # Check for quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Update display
                pygame.display.update()
                
                # Use the high FPS setting
                game.clock.tick(game.fps)
                game.game_counter += 1
                
                # Auto-close when game is over
                if game.winner is not None and game.game_counter > game_end_time:
                    running = False
            
            # Auto-close pygame when done
            pygame.quit()
            
            # Output results
            results = extract_game_results(game)
            print("GAME_RESULTS: " + json.dumps(results))
        
        # Replace the game's run method with our modified version
        game.run = modified_run
        game.run()
