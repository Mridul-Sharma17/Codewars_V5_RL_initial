"""
Strategy encoder/decoder for tower defense strategies.
Converts between parameter dictionaries and compact team_signal strings.
"""

from typing import Dict, Any, List, Optional

class StrategyEncoder:
    """
    Encodes strategy parameters into compact team_signal format and back.
    Enables efficient parameter passing between evolution system and game.
    """
    
    def __init__(self, max_length=200):
        """
        Initialize the encoder with a maximum length constraint.
        
        Args:
            max_length: Maximum length for encoded strings
        """
        self.max_length = max_length
        
        # Define troop mappings for compact encoding
        self.troop_mapping = {
            "dragon": "d",
            "wizard": "w",
            "valkyrie": "v",
            "musketeer": "m",
            "knight": "k",
            "archer": "a",
            "minion": "i",  # Use 'i' because 'm' is already used
            "barbarian": "b",
            "prince": "p",
            "giant": "g",
            "skeleton": "s",
            "balloon": "l"
        }
        
        # Reverse mapping for decoding
        self.reverse_troop_mapping = {v: k for k, v in self.troop_mapping.items()}
    
    def encode(self, strategy: Dict[str, Any]) -> str:
        """
        Encode strategy parameters as a compact string for team_signal.
        
        Args:
            strategy: Strategy parameter dictionary
            
        Returns:
            Encoded parameter string
        """
        # Create encoded parts
        parts = []
        
        # Version
        parts.append("v:1")
        
        # Lane preference
        lane_pref = strategy.get("lane_preference", "adaptive")
        if lane_pref:
            parts.append(f"l:{lane_pref[0]}")
        
        # Elixir thresholds
        elixir_thresholds = strategy.get("elixir_thresholds", {})
        min_deploy = elixir_thresholds.get("min_deploy", 4.0)
        parts.append(f"e:{min_deploy:.2f}")
        
        save_threshold = elixir_thresholds.get("save_threshold", 7.0)
        parts.append(f"s:{save_threshold:.2f}")
        
        emergency_threshold = elixir_thresholds.get("emergency_threshold", 2.0)
        parts.append(f"em:{emergency_threshold:.2f}")
        
        # Defensive trigger
        defensive_trigger = strategy.get("defensive_trigger", 0.65)
        parts.append(f"dt:{defensive_trigger:.2f}")
        
        aggression_trigger = strategy.get("aggression_trigger", 0.25)
        if aggression_trigger:
            parts.append(f"at:{aggression_trigger:.2f}")
        
        # Position settings
        position_settings = strategy.get("position_settings", {})
        y_default = position_settings.get("y_default", 40) 
        parts.append(f"y:{y_default}")
        
        y_defensive = position_settings.get("defensive_y", 20)
        parts.append(f"yd:{y_defensive}")
        
        y_aggressive = position_settings.get("y_aggressive", 47)
        if y_aggressive:
            parts.append(f"ya:{y_aggressive}")
        
        x_range = position_settings.get("x_range", [-20, 20])
        if isinstance(x_range, list) and len(x_range) >= 2:
            parts.append(f"xl:{x_range[0]}")
            parts.append(f"xr:{x_range[1]}")
        
        # Troop priority as abbreviations
        troop_priority = strategy.get("troop_priority", [])
        if troop_priority:
            abbr = ""
            for troop in troop_priority[:8]:  # Limit to 8 troops
                if not troop:
                    abbr += "x"
                    continue
                    
                # Get first letter of troop name
                if troop.lower() in self.troop_mapping:
                    abbr += self.troop_mapping[troop.lower()]
                else:
                    # Fallback to first letter if not in mapping
                    abbr += troop[0].lower()
                    
            parts.append(f"tp:{abbr}")
        
        # Counter weights
        counter_weights = strategy.get("counter_weights", {})
        
        air_vs_ground = counter_weights.get("air_vs_ground", 1.5)
        parts.append(f"ca:{air_vs_ground:.1f}")
        
        splash_vs_group = counter_weights.get("splash_vs_group", 2.0)
        parts.append(f"cs:{splash_vs_group:.1f}")
        
        tank_priority = counter_weights.get("tank_priority", 1.3)
        parts.append(f"ct:{tank_priority:.1f}")
        
        range_bonus = counter_weights.get("range_bonus", 1.5)
        parts.append(f"cr:{range_bonus:.1f}")
        
        # Join and ensure within max length
        signal = ",".join(parts)
        if len(signal) > self.max_length:
            signal = signal[:self.max_length]
            
        return signal
        
    def decode(self, signal: str) -> Dict[str, Any]:
        """
        Decode a strategy from encoded team_signal string.
        
        Args:
            signal: Encoded parameter string
            
        Returns:
            Strategy parameter dictionary
        """
        # Default strategy structure
        strategy = {
            "elixir_thresholds": {
                "min_deploy": 4.0,
                "save_threshold": 7.0,
                "emergency_threshold": 2.0
            },
            "position_settings": {
                "x_range": [-20, 20],
                "y_default": 40,
                "defensive_y": 20,
                "y_aggressive": 47
            },
            "defensive_trigger": 0.65,
            "aggression_trigger": 0.25,
            "counter_weights": {
                "air_vs_ground": 1.5,
                "splash_vs_group": 2.0,
                "tank_priority": 1.3,
                "range_bonus": 1.5
            },
            "lane_preference": "adaptive",
            "troop_priority": ["dragon", "wizard", "valkyrie", "musketeer", 
                              "knight", "archer", "minion", "barbarian"]
        }
        
        # Parse signal parts
        parts = signal.split(",")
        
        for part in parts:
            if ":" not in part:
                continue
                
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            try:
                # Version
                if key == "v":
                    strategy["version"] = value
                
                # Lane preference
                elif key == "l":
                    if value == "l":
                        strategy["lane_preference"] = "left"
                    elif value == "r":
                        strategy["lane_preference"] = "right"
                    elif value == "c":
                        strategy["lane_preference"] = "center"
                    elif value == "s":
                        strategy["lane_preference"] = "split"
                    else:
                        strategy["lane_preference"] = "adaptive"
                
                # Elixir thresholds
                elif key == "e":
                    strategy["elixir_thresholds"]["min_deploy"] = float(value)
                elif key == "s":
                    strategy["elixir_thresholds"]["save_threshold"] = float(value)
                elif key == "em":
                    strategy["elixir_thresholds"]["emergency_threshold"] = float(value)
                
                # Triggers
                elif key == "dt":
                    strategy["defensive_trigger"] = float(value)
                elif key == "at":
                    strategy["aggression_trigger"] = float(value)
                
                # Position settings
                elif key == "y":
                    strategy["position_settings"]["y_default"] = int(float(value))
                elif key == "yd":
                    strategy["position_settings"]["defensive_y"] = int(float(value))
                elif key == "ya":
                    strategy["position_settings"]["y_aggressive"] = int(float(value))
                elif key == "xl":
                    strategy["position_settings"]["x_range"][0] = int(float(value))
                elif key == "xr":
                    strategy["position_settings"]["x_range"][1] = int(float(value))
                
                # Troop priority
                elif key == "tp":
                    troops = []
                    for char in value:
                        if char.lower() in self.reverse_troop_mapping:
                            troops.append(self.reverse_troop_mapping[char.lower()])
                        elif char.lower() == "x":
                            # Skip placeholder
                            pass
                    
                    # If we found any troops, update the priority list
                    if troops:
                        strategy["troop_priority"] = troops
                
                # Counter weights
                elif key == "ca":
                    strategy["counter_weights"]["air_vs_ground"] = float(value)
                elif key == "cs":
                    strategy["counter_weights"]["splash_vs_group"] = float(value)
                elif key == "ct":
                    strategy["counter_weights"]["tank_priority"] = float(value)
                elif key == "cr":
                    strategy["counter_weights"]["range_bonus"] = float(value)
                
            except (ValueError, IndexError):
                # Skip parameters with invalid values
                pass
        
        return strategy
    
    def strategy_to_display_string(self, strategy: Dict[str, Any]) -> str:
        """
        Convert a strategy to a human-readable display string.
        
        Args:
            strategy: Strategy parameter dictionary
            
        Returns:
            Human-readable description string
        """
        lines = []
        
        # Basic info
        name = strategy.get("name", "Unnamed")
        lines.append(f"Strategy: {name}")
        
        # Lane preference
        lane = strategy.get("lane_preference", "adaptive")
        lines.append(f"Lane: {lane}")
        
        # Elixir management
        elixir = strategy.get("elixir_thresholds", {})
        min_deploy = elixir.get("min_deploy", 4.0)
        save = elixir.get("save_threshold", 7.0)
        emerg = elixir.get("emergency_threshold", 2.0)
        lines.append(f"Elixir: deploy={min_deploy}, save={save}, emergency={emerg}")
        
        # Position settings
        pos = strategy.get("position_settings", {})
        x_range = pos.get("x_range", [-20, 20])
        y_default = pos.get("y_default", 40)
        y_def = pos.get("defensive_y", 20)
        lines.append(f"Position: x={x_range}, y={y_default}, defensive_y={y_def}")
        
        # Triggers
        def_trigger = strategy.get("defensive_trigger", 0.65)
        lines.append(f"Defensive trigger: {def_trigger}")
        
        # Troop priority
        troops = strategy.get("troop_priority", [])
        if troops:
            troop_str = ", ".join(troops[:8])
            lines.append(f"Troop priority: {troop_str}")
        
        # Counter weights
        cw = strategy.get("counter_weights", {})
        lines.append("Counter weights:")
        for k, v in cw.items():
            lines.append(f"  {k}: {v}")
        
        # Metrics if available
        metrics = strategy.get("metrics", {})
        if metrics:
            games = metrics.get("games_played", 0)
            wins = metrics.get("wins", 0)
            win_rate = metrics.get("win_rate", 0)
            lines.append(f"Performance: {wins}/{games} wins ({win_rate:.3f})")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test encoder/decoder
    encoder = StrategyEncoder()
    
    # Example strategy
    test_strategy = {
        "name": "TestStrategy",
        "lane_preference": "right",
        "elixir_thresholds": {
            "min_deploy": 4.2,
            "save_threshold": 7.5,
            "emergency_threshold": 1.8
        },
        "position_settings": {
            "x_range": [-15, 15],
            "y_default": 42,
            "defensive_y": 18
        },
        "defensive_trigger": 0.7,
        "counter_weights": {
            "air_vs_ground": 1.8,
            "splash_vs_group": 2.1,
            "tank_priority": 1.5
        },
        "troop_priority": ["wizard", "dragon", "valkyrie", "musketeer", 
                          "knight", "archer", "minion", "barbarian"]
    }
    
    # Encode and decode
    encoded = encoder.encode(test_strategy)
    decoded = encoder.decode(encoded)
    
    print("Original:")
    print(encoder.strategy_to_display_string(test_strategy))
    print("\nEncoded:")
    print(encoded)
    print("\nDecoded:")
    print(encoder.strategy_to_display_string(decoded))