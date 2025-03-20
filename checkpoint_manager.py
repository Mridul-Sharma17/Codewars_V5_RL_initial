"""
Checkpoint manager for tower defense reinforcement learning system.
Handles saving and loading of training state to enable resuming from interruptions.
"""

import os
import json
import shutil
import glob
from datetime import datetime
from pathlib import Path
import time

class CheckpointManager:
    """
    Handles saving and loading of training checkpoints for the tower defense RL system.
    Enables resuming training from where it left off if interrupted.
    """
    
    def __init__(self, checkpoint_dir="training_data/checkpoints", max_checkpoints=3):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoint generations to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.latest_pointer_file = os.path.join(checkpoint_dir, "latest.json")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, generation, strategies, battle_metrics, training_state, interim=False):
        """
        Save a complete training checkpoint.
        
        Args:
            generation: Current generation number
            strategies: Dictionary of strategies
            battle_metrics: List of battle result metrics
            training_state: Dictionary with training state information
            interim: Whether this is an interim checkpoint within a generation
        
        Returns:
            Path to the saved checkpoint
        """
        # Create generation-specific checkpoint directory
        if interim:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}_interim")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}")
            
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Update timestamp in training state
        training_state["timestamp"] = datetime.now().isoformat()
        
        # Save all data components
        with open(os.path.join(checkpoint_path, "strategies.json"), 'w') as f:
            json.dump(strategies, f, indent=2)
            
        with open(os.path.join(checkpoint_path, "battle_metrics.json"), 'w') as f:
            json.dump(battle_metrics, f, indent=2)
            
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Create a metadata file with timestamp and type
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "type": "interim" if interim else "complete",
            "strategies_count": len(strategies),
            "battles_count": len(battle_metrics)
        }
        
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update the latest pointer file
        with open(self.latest_pointer_file, 'w') as f:
            json.dump({
                "path": checkpoint_path,
                "generation": generation,
                "timestamp": datetime.now().isoformat(),
                "type": "interim" if interim else "complete"
            }, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean up old checkpoints if we have more than max_checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def create_interim_checkpoint(self, generation, strategies, battle_metrics, 
                                training_state, battle_index, total_battles):
        """
        Create an interim checkpoint during a generation.
        
        Args:
            generation: Current generation number
            strategies: Dictionary of strategies
            battle_metrics: List of battle result metrics
            training_state: Dictionary with training state information
            battle_index: Current battle index
            total_battles: Total number of battles in this generation
            
        Returns:
            Path to the saved checkpoint
        """
        # Add interim-specific information to training state
        interim_state = training_state.copy()
        interim_state["interim_data"] = {
            "battle_index": battle_index,
            "total_battles": total_battles,
            "progress_percentage": round(battle_index / total_battles * 100, 2)
        }
        
        return self.save_checkpoint(
            generation, 
            strategies, 
            battle_metrics, 
            interim_state,
            interim=True
        )
    
    def load_latest_checkpoint(self):
        """
        Load the most recent checkpoint.
        
        Returns:
            Dictionary with loaded checkpoint data, or None if no checkpoint exists
        """
        if not os.path.exists(self.latest_pointer_file):
            print("No checkpoint found. Starting fresh training session.")
            return None
        
        try:
            with open(self.latest_pointer_file, 'r') as f:
                latest = json.load(f)
                
            checkpoint_path = latest["path"]
            
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint directory {checkpoint_path} not found.")
                return None
            
            # Load all components
            with open(os.path.join(checkpoint_path, "strategies.json"), 'r') as f:
                strategies = json.load(f)
                
            with open(os.path.join(checkpoint_path, "battle_metrics.json"), 'r') as f:
                battle_metrics = json.load(f)
                
            with open(os.path.join(checkpoint_path, "training_state.json"), 'r') as f:
                training_state = json.load(f)
            
            # Load metadata if available
            metadata_path = os.path.join(checkpoint_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "generation": latest["generation"],
                    "timestamp": latest["timestamp"],
                    "type": latest.get("type", "complete")
                }
            
            print(f"Loaded checkpoint from generation {latest['generation']}")
            print(f"Checkpoint timestamp: {metadata.get('timestamp', 'unknown')}")
            print(f"Checkpoint type: {metadata.get('type', 'complete')}")
            print(f"Strategies count: {len(strategies)}")
            print(f"Battles recorded: {len(battle_metrics)}")
            
            return {
                "generation": latest["generation"],
                "strategies": strategies,
                "battle_metrics": battle_metrics,
                "training_state": training_state,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def get_available_checkpoints(self):
        """
        Get a list of all available checkpoints.
        
        Returns:
            List of dictionaries with checkpoint information
        """
        checkpoint_dirs = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_gen_*"))
        checkpoints = []
        
        for checkpoint_dir in checkpoint_dirs:
            metadata_path = os.path.join(checkpoint_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata["path"] = checkpoint_dir
                    checkpoints.append(metadata)
            else:
                # Create basic metadata from directory name
                dir_name = os.path.basename(checkpoint_dir)
                generation = int(dir_name.replace("checkpoint_gen_", "").replace("_interim", ""))
                is_interim = "_interim" in dir_name
                
                metadata = {
                    "path": checkpoint_dir,
                    "generation": generation,
                    "type": "interim" if is_interim else "complete",
                    "timestamp": datetime.fromtimestamp(os.path.getctime(checkpoint_dir)).isoformat()
                }
                checkpoints.append(metadata)
        
        # Sort by generation, then by timestamp if available
        checkpoints.sort(key=lambda x: (x["generation"], x.get("timestamp", "")))
        
        return checkpoints
    
    def load_specific_checkpoint(self, generation):
        """
        Load a specific checkpoint by generation number.
        
        Args:
            generation: Generation number to load
            
        Returns:
            Dictionary with loaded checkpoint data, or None if not found
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}")
        
        if not os.path.exists(checkpoint_path):
            # Try interim checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}_interim")
            if not os.path.exists(checkpoint_path):
                print(f"No checkpoint found for generation {generation}")
                return None
        
        try:
            # Load all components
            with open(os.path.join(checkpoint_path, "strategies.json"), 'r') as f:
                strategies = json.load(f)
                
            with open(os.path.join(checkpoint_path, "battle_metrics.json"), 'r') as f:
                battle_metrics = json.load(f)
                
            with open(os.path.join(checkpoint_path, "training_state.json"), 'r') as f:
                training_state = json.load(f)
            
            # Load metadata if available
            metadata_path = os.path.join(checkpoint_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "generation": generation,
                    "timestamp": datetime.fromtimestamp(os.path.getctime(checkpoint_path)).isoformat(),
                    "type": "interim" if "_interim" in checkpoint_path else "complete"
                }
            
            print(f"Loaded checkpoint from generation {generation}")
            
            return {
                "generation": generation,
                "strategies": strategies,
                "battle_metrics": battle_metrics,
                "training_state": training_state,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints, keeping only the most recent ones."""
        # Get all non-interim checkpoints
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_gen_[0-9]*"))
        checkpoints = [cp for cp in checkpoints if "_interim" not in cp]
        
        # Sort by creation time (newest first)
        checkpoints.sort(key=lambda x: os.path.getctime(x), reverse=True)
        
        # Remove all but the last max_checkpoints
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[self.max_checkpoints:]:
                try:
                    shutil.rmtree(old_checkpoint)
                    print(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    print(f"Error removing old checkpoint {old_checkpoint}: {e}")
        
        # Always clean up interim checkpoints that aren't the most recent
        interim_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_gen_*_interim"))
        
        # Get the most recent checkpoint (interim or not) from the latest pointer
        if os.path.exists(self.latest_pointer_file):
            with open(self.latest_pointer_file, 'r') as f:
                latest = json.load(f)
                latest_path = latest["path"]
        else:
            latest_path = None
            
        for interim in interim_checkpoints:
            # Keep the interim checkpoint if it's the most recent one
            if interim != latest_path:
                try:
                    shutil.rmtree(interim)
                    print(f"Removed obsolete interim checkpoint: {interim}")
                except Exception as e:
                    print(f"Error removing interim checkpoint {interim}: {e}")


if __name__ == "__main__":
    # Example usage
    manager = CheckpointManager()
    
    # Check for available checkpoints
    checkpoints = manager.get_available_checkpoints()
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints:")
        for i, cp in enumerate(checkpoints):
            print(f"{i+1}. Generation {cp['generation']} ({cp['type']}) - {cp['timestamp']}")
    else:
        print("No checkpoints found.")
    
    # Example of creating a test checkpoint
    test_save = False
    if test_save:
        example_strategies = {"strategy1": {"name": "test1"}, "strategy2": {"name": "test2"}}
        example_metrics = [{"battle": 1, "winner": "test1"}]
        example_state = {"current_generation": 1, "best_strategy": "strategy1"}
        
        manager.save_checkpoint(1, example_strategies, example_metrics, example_state)
        print("Test checkpoint created.")
        
        # Load it back
        checkpoint = manager.load_latest_checkpoint()
        if checkpoint:
            print("Successfully loaded test checkpoint.")