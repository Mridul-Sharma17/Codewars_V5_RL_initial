import os
import sys
import importlib.util
import subprocess
import json
import time
import tempfile
import shutil
from pathlib import Path
import queue

class GameRunner:
    """Handles running games between different strategies and collecting results"""
    
    def __init__(self, game_path="game", headless=True, speed_multiplier=100000):
        """
        Initialize the game runner
        
        Args:
            game_path: Path to the game directory
            headless: Whether to run in headless mode (no graphics)
            speed_multiplier: How much to speed up the game (higher = faster)
        """
        self.game_path = game_path
        self.headless = headless
        self.speed_multiplier = speed_multiplier
        self.result_queue = queue.Queue()
        
        # Ensure game directory exists
        if not os.path.exists(game_path):
            raise ValueError(f"Game directory not found: {game_path}")
            
        # Check for required game files
        required_files = ["main.py", "game.py", "config.py"]
        for file in required_files:
            if not os.path.exists(os.path.join(game_path, file)):
                raise ValueError(f"Required game file not found: {file}")
        
        # Create backup of original config file
        config_path = os.path.join(game_path, "config.py")
        backup_path = os.path.join(game_path, "config_backup.py")
        if not os.path.exists(backup_path):
            shutil.copy(config_path, backup_path)
            print(f"Created backup of original config file: {backup_path}")
    
    def prepare_game(self, team1_path, team2_path):
        """
        Prepare the game for a battle between two teams
        
        Args:
            team1_path: Path to team 1's script
            team2_path: Path to team 2's script
        """
        # Copy team files to the game's teams directory
        teams_dir = os.path.join(self.game_path, "teams")
        if not os.path.exists(teams_dir):
            os.makedirs(teams_dir)
            
        # Copy helper_function.py if it doesn't already exist in teams directory
        helper_path = os.path.join(teams_dir, "helper_function.py")
        if not os.path.exists(helper_path):
            # Look for helper_function.py in the current directory or parent directories
            for dir_path in [os.path.dirname(team1_path), ".", ".."]:
                source_path = os.path.join(dir_path, "helper_function.py")
                if os.path.exists(source_path):
                    shutil.copy(source_path, helper_path)
                    print(f"Copied helper_function.py to {helper_path}")
                    break
                    
        # Copy team scripts to a.py and b.py
        shutil.copy(team1_path, os.path.join(teams_dir, "a.py"))
        shutil.copy(team2_path, os.path.join(teams_dir, "b.py"))
        print(f"Team files copied: {os.path.basename(team1_path)} -> a.py, {os.path.basename(team2_path)} -> b.py")
        
        # Update config.py to use these teams
        self._update_config()
        
    def _update_config(self):
        """Update the game's config.py file to use our teams"""
        config_path = os.path.join(self.game_path, "config.py")
        with open(config_path, 'w') as f:
            f.write("from teams import a,b\n\n")
            f.write("TEAM1 = a\n")
            f.write("TEAM2 = b\n")
            f.write("VALUE_ERROR = False\n")
            # Add headless mode indicator
            if self.headless:
                f.write("HEADLESS_MODE = True\n")
        print("Updated config.py to use our teams")
        
    def _restore_config(self):
        """Restore the original config.py from backup"""
        backup_path = os.path.join(self.game_path, "config_backup.py")
        config_path = os.path.join(self.game_path, "config.py")
        if os.path.exists(backup_path):
            shutil.copy(backup_path, config_path)
            print("Restored original config.py")
    
    def _run_game_process(self):
        """Run the game in a separate process and monitor for results"""
        # Create a modified version of main.py that will output results
        self._create_instrumented_main()
        
        # Change to game directory
        original_dir = os.getcwd()
        os.chdir(self.game_path)
        
        try:
            # Set environment variables for headless mode
            env = os.environ.copy()
            if self.headless:
                # This completely disables all display functionality
                env["SDL_VIDEODRIVER"] = "dummy"
                env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
                # Additional env vars to ensure no display is used
                env["SDL_AUDIODRIVER"] = "dummy"
                
            # Run the instrumented main.py
            cmd = [sys.executable, "instrumented_main.py"]
            
            # Start the process with modified environment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            # Check for errors
            if process.returncode != 0:
                print(f"Game process failed with return code {process.returncode}")
                print(f"Error: {stderr}")
                return None
                
            # Parse the results from stdout
            result = self._parse_results(stdout)
            return result
            
        finally:
            # Clean up
            if os.path.exists("instrumented_main.py"):
                os.remove("instrumented_main.py")
            os.chdir(original_dir)
    
    def _create_instrumented_main(self):
        """Create a modified main.py that outputs game results"""
        # Read the original main.py
        with open(os.path.join(self.game_path, "main.py"), 'r') as f:
            main_code = f.read()
            
        # Create the instrumented version
        with open(os.path.join(self.game_path, "instrumented_main.py"), 'w') as f:
            # Add imports and headless mode setup at the top
            f.write("import os\n")
            f.write("import json\n")
            f.write("import sys\n")
            f.write("import time\n\n")
            
            # Set up headless mode by configuring pygame before import
            if self.headless:
                f.write("# Configure pygame for headless operation before import\n")
                f.write("os.environ['SDL_VIDEODRIVER'] = 'dummy'\n")
                f.write("os.environ['SDL_AUDIODRIVER'] = 'dummy'\n")
                f.write("os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'\n\n")
            
            f.write("from scripts.game_config import *\n\n")
            
            # Add our game result extraction function
            f.write("def extract_game_results(game):\n")
            f.write("    \"\"\"Extract results from the game object\"\"\"\n")
            f.write("    results = {\n")
            f.write("        \"winner\": game.winner,\n")
            f.write("        \"team1_name\": game.team_name1,\n")
            f.write("        \"team2_name\": game.team_name2,\n")
            f.write("        \"team1_health\": game.tower1.health if hasattr(game, \"tower1\") else 0,\n")
            f.write("        \"team2_health\": game.tower2.health if hasattr(game, \"tower2\") else 0,\n")
            f.write("        \"game_duration\": game.game_counter,\n")
            f.write("        \"message\": game.message\n")
            f.write("    }\n")
            f.write("    return results\n\n")
            
            # Add the headless game simulation function
            if self.headless:
                f.write("def headless_simulation(game):\n")
                f.write("    \"\"\"Run the game simulation without any rendering\"\"\"\n")
                f.write("    # Set game speed very high for fastest simulation\n")
                f.write("    game.fps = 100000\n\n")
                f.write("    # Run game for a maximum number of frames\n")
                f.write("    game_end_time = 1830  # Default game end time\n")
                f.write("    if 'GAME_END_TIME' in globals():\n")
                f.write("        game_end_time = GAME_END_TIME\n\n")
                f.write("    # Run simulation loop without any rendering\n")
                f.write("    max_frames = game_end_time + 100  # Add buffer for outro\n")
                f.write("    while game.game_counter < max_frames:\n")
                f.write("        # Only update game mechanics, no rendering\n")
                f.write("        game.update_game_mechanics()\n")
                f.write("        game.game_counter += 1\n\n")
                f.write("        # Check if game has ended\n")
                f.write("        if game.winner is not None and game.game_counter > game_end_time:\n")
                f.write("            break\n\n")
                f.write("    # Output results\n")
                f.write("    results = extract_game_results(game)\n")
                f.write("    print(\"GAME_RESULTS: \" + json.dumps(results))\n\n")
            
            # Modify the end of the original main code to use our headless simulation if enabled
            modified_code = main_code.replace(
                "if team1_test_pass and team2_test_pass:\n    Game(TEAM1.troops,TEAM2.troops,TEAM1.team_name,TEAM2.team_name).run()",
                """
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
        # Regular display mode - override run method
        original_run = game.run
        def modified_run():
            # Set game speed higher
            game.fps = 1000
            
            # Run game for a maximum number of frames
            game_end_time = 1830  # Default
            if 'GAME_END_TIME' in globals():
                game_end_time = GAME_END_TIME
                
            # Run with visual rendering
            max_frames = game_end_time + 100  # Add buffer for outro
            while game.game_counter < max_frames:
                game.render_game_screen()
                game.render_left_screen()
                game.render_right_screen()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                        
                pygame.display.update()
                
                # Update game state
                game.clock.tick(game.fps)
                game.game_counter += 1
                
                # Check if game has ended
                if game.winner is not None and game.game_counter > game_end_time:
                    break
                    
            # Output results
            results = extract_game_results(game)
            print("GAME_RESULTS: " + json.dumps(results))
            pygame.quit()
        
        game.run = modified_run
        game.run()
"""
            )
            
            f.write(modified_code)
    
    def _parse_results(self, stdout):
        """Parse game results from the stdout of the game process"""
        for line in stdout.splitlines():
            if line.startswith("GAME_RESULTS: "):
                # Extract the JSON part
                json_str = line[len("GAME_RESULTS: "):]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Failed to decode game results: {json_str}")
                    return None
        
        print("No game results found in output")
        return None
    
    def run_battle(self, team1_path, team2_path):
        """
        Run a battle between two teams and return the results
        
        Args:
            team1_path: Path to team 1's script
            team2_path: Path to team 2's script
            
        Returns:
            Dictionary with battle results
        """
        try:
            # Prepare the game
            self.prepare_game(team1_path, team2_path)
            
            # Run the game
            results = self._run_game_process()
            
            if results:
                # Add additional metadata
                results["team1_file"] = os.path.basename(team1_path)
                results["team2_file"] = os.path.basename(team2_path)
                results["timestamp"] = time.time()
                
                team1_win = results["winner"] == results["team1_name"]
                team2_win = results["winner"] == results["team2_name"]
                
                results["team1_win"] = 1 if team1_win else 0
                results["team2_win"] = 1 if team2_win else 0
                
                print(f"Battle completed: {results['team1_name']} vs {results['team2_name']}")
                print(f"Winner: {results['winner']}")
                
            return results
            
        finally:
            # Clean up
            self._restore_config()
    
    def run_tournament(self, team_files, matches_per_pair=1, output_file=None):
        """
        Run a tournament between multiple teams
        
        Args:
            team_files: List of paths to team script files
            matches_per_pair: Number of matches to run for each pair of teams
            output_file: Optional file to save tournament results
            
        Returns:
            List of results for all matches
        """
        all_results = []
        
        # Generate all pairs of teams
        team_pairs = []
        for i, team1 in enumerate(team_files):
            for j, team2 in enumerate(team_files):
                if i != j:  # Don't match a team against itself
                    for _ in range(matches_per_pair):
                        team_pairs.append((team1, team2))
        
        print(f"Running tournament with {len(team_pairs)} matches")
        
        # Run battles for each pair
        for i, (team1, team2) in enumerate(team_pairs):
            print(f"Match {i+1}/{len(team_pairs)}: {os.path.basename(team1)} vs {os.path.basename(team2)}")
            result = self.run_battle(team1, team2)
            if result:
                all_results.append(result)
                
            # Save intermediate results
            if output_file and all_results:
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
        
        return all_results