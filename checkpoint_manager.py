"""
Tower Defense RL - Checkpoint Manager

This module handles saving and loading checkpoints during training.
"""

import os
import json
import glob
import shutil
import logging
from typing import Dict, Any, Optional, List
import datetime

from config import *

logger = logging.getLogger('checkpoint_manager')

class CheckpointManager:
    """Handles saving and loading of checkpoints during training."""
    
    def __init__(self):
        """Initialize the checkpoint manager."""
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoints_to_keep = CHECKPOINTS_TO_KEEP
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> str:
        """
        Save a checkpoint.
        
        Args:
            checkpoint_data: Data to save in the checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        generation = checkpoint_data.get("generation", 0)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}")
        
        # Create checkpoint directory
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save checkpoint data
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        
        # Save the "latest" reference
        latest_file = os.path.join(self.checkpoint_dir, "latest.json")
        with open(latest_file, 'w') as f:
            json.dump({
                "generation": generation,
                "path": checkpoint_path,
                "timestamp": datetime.datetime.now().isoformat()
            }, f, indent=4)
        
        # Prune old checkpoints if needed
        self._prune_old_checkpoints()
        
        logger.info(f"Saved checkpoint for generation {generation} to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, generation: int) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for a specific generation.
        
        Args:
            generation: Generation number to load
            
        Returns:
            Checkpoint data, or None if checkpoint not found
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}")
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.json")
        
        if not os.path.exists(checkpoint_file):
            logger.warning(f"Checkpoint for generation {generation} not found")
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Loaded checkpoint for generation {generation} from {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Error loading checkpoint for generation {generation}: {e}")
            return None
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint.
        
        Returns:
            Latest checkpoint data, or None if no checkpoint found
        """
        latest_file = os.path.join(self.checkpoint_dir, "latest.json")
        
        if not os.path.exists(latest_file):
            # Try to find the latest checkpoint manually
            checkpoint_dirs = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_gen_*"))
            
            if not checkpoint_dirs:
                logger.warning("No checkpoints found")
                return None
            
            # Extract generation numbers and find the latest
            generations = []
            for dir_path in checkpoint_dirs:
                try:
                    gen_num = int(dir_path.split("_")[-1])
                    generations.append((gen_num, dir_path))
                except (ValueError, IndexError):
                    continue
            
            if not generations:
                logger.warning("No valid checkpoints found")
                return None
            
            latest_gen, latest_path = max(generations, key=lambda x: x[0])
            logger.info(f"Found latest checkpoint for generation {latest_gen}")
            
            return self.load_checkpoint(latest_gen)
        
        try:
            with open(latest_file, 'r') as f:
                latest_info = json.load(f)
            
            generation = latest_info.get("generation")
            return self.load_checkpoint(generation)
        except Exception as e:
            logger.error(f"Error loading latest checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        checkpoint_dirs = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_gen_*"))
        checkpoints = []
        
        for dir_path in checkpoint_dirs:
            try:
                gen_num = int(dir_path.split("_")[-1])
                checkpoint_file = os.path.join(dir_path, "checkpoint.json")
                
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        info = json.load(f)
                    
                    checkpoints.append({
                        "generation": gen_num,
                        "path": dir_path,
                        "timestamp": info.get("timestamp")
                    })
            except (ValueError, IndexError, FileNotFoundError, json.JSONDecodeError):
                continue
        
        # Sort by generation
        checkpoints.sort(key=lambda x: x["generation"])
        return checkpoints
    
    def _prune_old_checkpoints(self) -> None:
        """Prune old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.checkpoints_to_keep:
            return
        
        # Keep the most recent checkpoints
        to_delete = checkpoints[:-self.checkpoints_to_keep]
        
        for checkpoint in to_delete:
            path = checkpoint["path"]
            try:
                shutil.rmtree(path)
                logger.info(f"Deleted old checkpoint: {path}")
            except Exception as e:
                logger.error(f"Error deleting checkpoint {path}: {e}")