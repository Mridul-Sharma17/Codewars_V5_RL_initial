import re
import json
from typing import Dict, Any, List, Tuple, Optional

class StrategyEncoder:
    """
    Handles encoding and decoding of strategy parameters to fit within the team_signal
    character limit (200 chars) while preserving key strategy information.
    """
    
    def __init__(self, max_length: int = 200):
        """
        Initialize the strategy encoder.
        
        Args:
            max_length: Maximum length for encoded strategy string
        """
        self.max_length = max_length
        
        # Define parameter keys and their short codes
        self.param_codes = {
            "version": "v",
            "lane_preference": "l",
            "elixir_thresholds.min_deploy": "e",
            "elixir_thresholds.save_threshold": "s",
            "elixir_thresholds.emergency_threshold": "m",
            "position_settings.y_default": "y",
            "position_settings.defensive_y": "d",
            "position_settings.x_range.left": "xl",
            "position_settings.x_range.right": "xr",
            "defensive_trigger": "dt",
            "counter_weights.air_vs_ground": "ca",
            "counter_weights.splash_vs_group": "cs",
            "counter_weights.tank_priority": "ct",
            "troop_priority": "tp",
            "wins": "w",
            "losses": "L"
        }
        
        # Reverse lookup for decoding
        self.code_params = {v: k for k, v in self.param_codes.items()}
        
        # Lane preference shortenings
        self.lane_codes = {
            "left": "L",
            "right": "R",
            "center": "C",
            "split": "S"
        }
        
        # Reverse lookup for lane codes
        self.lane_decode = {v: k for k, v in self.lane_codes.items()}
        
        # Troop name abbreviations (first letter of each troop)
        self.troop_abbrevs = {
            "archer": "a",
            "giant": "g",
            "dragon": "d",
            "balloon": "b",
            "prince": "p",
            "barbarian": "B",
            "knight": "k",
            "minion": "m",
            "skeleton": "s",
            "wizard": "w",
            "valkyrie": "v",
            "musketeer": "M"
        }
        
        # Reverse lookup for troop abbreviations
        self.troop_decode = {v: k for k, v in self.troop_abbrevs.items()}
    
    def encode(self, strategy: Dict[str, Any]) -> str:
        """
        Encode a strategy into a compact string representation.
        
        Args:
            strategy: Strategy dictionary
            
        Returns:
            Compact string representation
        """
        parts = []
        
        # Version always comes first
        parts.append(f"{self.param_codes['version']}:1")
        
        # Add lane preference with short code
        if "lane_preference" in strategy:
            lane = strategy["lane_preference"]
            lane_code = self.lane_codes.get(lane, lane[0].upper())
            parts.append(f"{self.param_codes['lane_preference']}:{lane_code}")
        
        # Add elixir thresholds
        if "elixir_thresholds" in strategy:
            et = strategy["elixir_thresholds"]
            if "min_deploy" in et:
                parts.append(f"{self.param_codes['elixir_thresholds.min_deploy']}:{et['min_deploy']}")
            if "save_threshold" in et:
                parts.append(f"{self.param_codes['elixir_thresholds.save_threshold']}:{et['save_threshold']}")
            if "emergency_threshold" in et:
                parts.append(f"{self.param_codes['elixir_thresholds.emergency_threshold']}:{et['emergency_threshold']}")
        
        # Add position settings
        if "position_settings" in strategy:
            ps = strategy["position_settings"]
            if "y_default" in ps:
                parts.append(f"{self.param_codes['position_settings.y_default']}:{ps['y_default']}")
            if "defensive_y" in ps:
                parts.append(f"{self.param_codes['position_settings.defensive_y']}:{ps['defensive_y']}")
            if "x_range" in ps and isinstance(ps["x_range"], list) and len(ps["x_range"]) >= 2:
                parts.append(f"{self.param_codes['position_settings.x_range.left']}:{ps['x_range'][0]}")
                parts.append(f"{self.param_codes['position_settings.x_range.right']}:{ps['x_range'][1]}")
        
        # Add defensive trigger
        if "defensive_trigger" in strategy:
            parts.append(f"{self.param_codes['defensive_trigger']}:{strategy['defensive_trigger']:.1f}")
        
        # Add counter weights
        if "counter_weights" in strategy:
            cw = strategy["counter_weights"]
            if "air_vs_ground" in cw:
                parts.append(f"{self.param_codes['counter_weights.air_vs_ground']}:{cw['air_vs_ground']:.1f}")
            if "splash_vs_group" in cw:
                parts.append(f"{self.param_codes['counter_weights.splash_vs_group']}:{cw['splash_vs_group']:.1f}")
            if "tank_priority" in cw:
                parts.append(f"{self.param_codes['counter_weights.tank_priority']}:{cw['tank_priority']:.1f}")
        
        # Add troop priority as abbreviated sequence
        if "troop_priority" in strategy:
            abbrevs = []
            for troop in strategy["troop_priority"]:
                abbrev = self.troop_abbrevs.get(troop, troop[0].lower())
                abbrevs.append(abbrev)
            if abbrevs:
                parts.append(f"{self.param_codes['troop_priority']}:{''.join(abbrevs)}")
        
        # Add wins/losses from metrics if available
        if "metrics" in strategy:
            metrics = strategy["metrics"]
            if "wins" in metrics and metrics["wins"] > 0:
                parts.append(f"{self.param_codes['wins']}:{metrics['wins']}")
            if "losses" in metrics and metrics["losses"] > 0:
                parts.append(f"{self.param_codes['losses']}:{metrics['losses']}")
        
        # Combine all parts with commas
        encoded = ",".join(parts)
        
        # Ensure we don't exceed the maximum length
        if len(encoded) > self.max_length:
            # Priority order: version, lane, elixir, position, troops, metrics
            # Start removing the least important parameters
            while len(encoded) > self.max_length:
                # Try removing weights first
                if any(p in encoded for p in ["ca:", "cs:", "ct:"]):
                    for code in ["ct:", "cs:", "ca:"]:
                        pattern = f",{code}[^,]*"
                        match = re.search(pattern, encoded)
                        if match:
                            encoded = encoded.replace(match.group(0), "")
                            break
                    continue
                
                # Then try removing losses
                if "L:" in encoded:
                    pattern = f",{self.param_codes['losses']}:[^,]*"
                    match = re.search(pattern, encoded)
                    if match:
                        encoded = encoded.replace(match.group(0), "")
                        continue
                
                # Then try removing wins
                if "w:" in encoded:
                    pattern = f",{self.param_codes['wins']}:[^,]*"
                    match = re.search(pattern, encoded)
                    if match:
                        encoded = encoded.replace(match.group(0), "")
                        continue
                
                # Then try shortening troop priority
                if "tp:" in encoded:
                    pattern = f"{self.param_codes['troop_priority']}:([^,]*)"
                    match = re.search(pattern, encoded)
                    if match:
                        troops = match.group(1)
                        if len(troops) > 4:
                            shortened = troops[:4]  # Keep only first 4 troops
                            encoded = encoded.replace(f"{self.param_codes['troop_priority']}:{troops}", 
                                                     f"{self.param_codes['troop_priority']}:{shortened}")
                            continue
                
                # Last resort, truncate and add ellipsis
                if len(encoded) > self.max_length:
                    encoded = encoded[:self.max_length-3] + "..."
                    break
        
        return encoded
    
    def decode(self, signal: str) -> Dict[str, Any]:
        """
        Decode a strategy from its string representation.
        
        Args:
            signal: Encoded strategy string
            
        Returns:
            Strategy dictionary with decoded parameters
        """
        if not signal or ":" not in signal:
            # Return empty strategy if signal is invalid
            return {}
        
        strategy = {
            "elixir_thresholds": {},
            "position_settings": {},
            "counter_weights": {}
        }
        
        # Split into parts
        parts = signal.split(",")
        
        for part in parts:
            if ":" not in part:
                continue
                
            code, value = part.split(":", 1)
            code = code.strip()
            value = value.strip()
            
            if code not in self.code_params:
                # Unknown code, skip
                continue
                
            param = self.code_params[code]
            
            # Handle specific parameter types
            if param == "version":
                strategy["version"] = int(value)
                
            elif param == "lane_preference":
                if value in self.lane_decode:
                    strategy["lane_preference"] = self.lane_decode[value]
                else:
                    # Default to first letter as lane name
                    lanes = {"L": "left", "R": "right", "C": "center", "S": "split"}
                    strategy["lane_preference"] = lanes.get(value, "right")
                    
            elif param.startswith("elixir_thresholds."):
                subparam = param.split(".", 1)[1]
                try:
                    strategy["elixir_thresholds"][subparam] = float(value)
                except ValueError:
                    strategy["elixir_thresholds"][subparam] = 4  # Default
                    
            elif param.startswith("position_settings."):
                parts = param.split(".", 1)[1]
                if parts == "x_range.left":
                    if "x_range" not in strategy["position_settings"]:
                        strategy["position_settings"]["x_range"] = [-20, 20]  # Default
                    try:
                        strategy["position_settings"]["x_range"][0] = int(value)
                    except ValueError:
                        pass
                elif parts == "x_range.right":
                    if "x_range" not in strategy["position_settings"]:
                        strategy["position_settings"]["x_range"] = [-20, 20]  # Default
                    try:
                        strategy["position_settings"]["x_range"][1] = int(value)
                    except ValueError:
                        pass
                else:
                    try:
                        strategy["position_settings"][parts] = int(value)
                    except ValueError:
                        if parts == "y_default":
                            strategy["position_settings"][parts] = 45  # Default
                        elif parts == "defensive_y":
                            strategy["position_settings"][parts] = 20  # Default
                    
            elif param == "defensive_trigger":
                try:
                    strategy["defensive_trigger"] = float(value)
                except ValueError:
                    strategy["defensive_trigger"] = 0.7  # Default
                    
            elif param.startswith("counter_weights."):
                subparam = param.split(".", 1)[1]
                try:
                    strategy["counter_weights"][subparam] = float(value)
                except ValueError:
                    defaults = {
                        "air_vs_ground": 1.2,
                        "splash_vs_group": 1.5,
                        "tank_priority": 1.3
                    }
                    strategy["counter_weights"][subparam] = defaults.get(subparam, 1.0)
                    
            elif param == "troop_priority":
                # Decode abbreviated troops
                troops = []
                for char in value:
                    if char in self.troop_decode:
                        troops.append(self.troop_decode[char])
                    else:
                        # Try to guess based on first letter
                        found = False
                        for troop, abbrev in self.troop_abbrevs.items():
                            if troop[0].lower() == char.lower():
                                troops.append(troop)
                                found = True
                                break
                        if not found and char.isalpha():
                            # Use as-is if we can't match
                            troops.append(char.lower())
                
                strategy["troop_priority"] = troops
                
            elif param in ["wins", "losses"]:
                if "metrics" not in strategy:
                    strategy["metrics"] = {}
                try:
                    strategy["metrics"][param] = int(value)
                except ValueError:
                    strategy["metrics"][param] = 0
        
        return strategy
    
    def extract_opponent_troops(self, signal: str) -> List[str]:
        """
        Extract opponent troop names from a signal string.
        These are typically added during gameplay to track opponent's deck.
        
        Args:
            signal: Encoded strategy string
            
        Returns:
            List of opponent troop names
        """
        if not signal:
            return []
            
        # Strategy parameters use key:value format
        # Opponent troop names are added as plain strings
        troops = []
        
        parts = signal.split(",")
        for part in parts:
            part = part.strip()
            # If there's no colon, it might be a troop name
            if ":" not in part and part and not part.endswith("..."):
                troops.append(part)
        
        return troops

    def update_signal_with_troops(self, signal: str, new_troops: List[str]) -> str:
        """
        Update signal with new opponent troops while preserving strategy parameters.
        
        Args:
            signal: Current signal string
            new_troops: New troop names to add
            
        Returns:
            Updated signal string
        """
        # First extract all the parameter parts (containing :)
        param_parts = []
        existing_troops = []
        
        if signal:
            for part in signal.split(","):
                part = part.strip()
                if ":" in part:
                    param_parts.append(part)
                elif part:  # Non-empty troop name
                    existing_troops.append(part)
        
        # Combine existing and new troops, removing duplicates
        all_troops = existing_troops.copy()
        for troop in new_troops:
            if troop not in all_troops:
                all_troops.append(troop)
        
        # Reconstruct signal
        new_signal = ",".join(param_parts)
        if new_signal and all_troops:
            new_signal += ","
        if all_troops:
            new_signal += ",".join(all_troops)
        
        # Ensure it doesn't exceed max length
        if len(new_signal) > self.max_length:
            # First try to limit the number of troops
            param_str = ",".join(param_parts)
            max_troops_length = self.max_length - len(param_str) - (1 if param_str else 0)
            
            if max_troops_length > 0:
                troops_str = ""
                for troop in all_troops:
                    if len(troops_str) + len(troop) + 1 <= max_troops_length:
                        if troops_str:
                            troops_str += ","
                        troops_str += troop
                    else:
                        break
                
                new_signal = param_str
                if param_str and troops_str:
                    new_signal += ","
                new_signal += troops_str
            else:
                # If we can't fit any troops, prioritize parameters
                new_signal = param_str
                if len(new_signal) > self.max_length:
                    new_signal = new_signal[:self.max_length-3] + "..."
        
        return new_signal


if __name__ == "__main__":
    # Example usage
    encoder = StrategyEncoder()
    
    test_strategy = {
        "lane_preference": "right",
        "elixir_thresholds": {
            "min_deploy": 3.5,
            "save_threshold": 7,
            "emergency_threshold": 2
        },
        "position_settings": {
            "x_range": [-20, 20],
            "y_default": 45,
            "defensive_y": 20
        },
        "defensive_trigger": 0.7,
        "counter_weights": {
            "air_vs_ground": 1.5,
            "splash_vs_group": 1.7,
            "tank_priority": 1.2
        },
        "troop_priority": ["dragon", "wizard", "valkyrie", "musketeer", "knight", "archer", "minion", "barbarian"],
        "metrics": {
            "games_played": 10,
            "wins": 8,
            "losses": 2,
            "win_rate": 0.8
        }
    }
    
    # Test encoding
    encoded = encoder.encode(test_strategy)
    print(f"Encoded strategy ({len(encoded)} chars):")
    print(encoded)
    
    # Test decoding
    decoded = encoder.decode(encoded)
    print("\nDecoded strategy:")
    print(json.dumps(decoded, indent=2))
    
    # Test updating with troops
    updated = encoder.update_signal_with_troops(encoded, ["Knight", "Archer", "Wizard"])
    print(f"\nUpdated with troops ({len(updated)} chars):")
    print(updated)
    
    # Test extracting troops
    troops = encoder.extract_opponent_troops(updated)
    print("\nExtracted troops:")
    print(troops)