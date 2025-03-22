"""
Tower Defense RL - Adaptive Templates

This module generates strategy code based on templates and parameters.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from config import *

logger = logging.getLogger('adaptive_templates')

class AdaptiveTemplates:
    """Generates strategy code based on templates and parameters."""
    
    def __init__(self):
        """Initialize the adaptive templates."""
        self.template_dir = TEMPLATE_DIR
        
        # Ensure template directory exists
        os.makedirs(self.template_dir, exist_ok=True)
    
    def generate_strategy_code(
        self, 
        template_type: str, 
        parameters: Dict[str, Any],
        strategy_name: str
    ) -> str:
        """
        Generate strategy code based on template and parameters.
        
        Args:
            template_type: Type of template to use
            parameters: Parameters to use with the template
            strategy_name: Name of the strategy
            
        Returns:
            Generated strategy code
        """
        if template_type == "direct_code" and "source_file" in parameters:
            # Simply load and return the exact file content
            with open(parameters["source_file"], 'r') as f:
                return f.read()
        
        # Load the template file
        template_path = os.path.join(self.template_dir, f"{template_type}_template.py")
        
        if not os.path.exists(template_path):
            logger.warning(f"Template file {template_path} not found. Using base template.")
            template_path = os.path.join(self.template_dir, "base_template.py")
        
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Replace template variables with parameter values
        code = template.replace('STRATEGY_NAME', strategy_name)
        
        # Generate parameter code
        param_dict_code = self._generate_parameter_dict(parameters)
        code = code.replace('PARAMETER_DICT', param_dict_code)
        
        # Select 8 troops
        troops = self._select_troops(template_type, parameters)
        troops_code = ', '.join(f'Troops.{troop.lower()}' for troop in troops)
        code = code.replace('TROOP_SELECTION', troops_code)
        
        return code
    
    def _generate_parameter_dict(self, parameters: Dict[str, Any]) -> str:
        """
        Generate Python code for parameter dictionary.
        
        Args:
            parameters: Parameters to convert to code
            
        Returns:
            Python code representing the parameters
        """
        lines = []
        lines.append("{")
        
        for key, value in parameters.items():
            if isinstance(value, dict):
                # Nested dictionary
                lines.append(f"    '{key}': {{")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        # Doubly-nested dictionary
                        lines.append(f"        '{subkey}': {{")
                        for subsubkey, subsubvalue in subvalue.items():
                            lines.append(f"            '{subsubkey}': {repr(subsubvalue)},")
                        lines.append("        },")
                    else:
                        lines.append(f"        '{subkey}': {repr(subvalue)},")
                lines.append("    },")
            else:
                lines.append(f"    '{key}': {repr(value)},")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _select_troops(self, template_type: str, parameters: Dict[str, Any]) -> List[str]:
        """
        Select 8 troops based on template type and parameters.
        
        Args:
            template_type: Type of template
            parameters: Parameters for the template
            
        Returns:
            List of 8 troop names
        """
        # All available troops
        all_troops = [
            "Archer", "Giant", "Dragon", "Balloon", "Prince", 
            "Barbarian", "Knight", "Minion", "Skeleton", 
            "Wizard", "Valkyrie", "Musketeer"
        ]
        
        if template_type == "counter_picker":
            # Counter picker needs a diverse set of troops for countering
            return self._select_diverse_troops(all_troops, parameters)
        elif template_type == "resource_manager":
            # Resource manager needs a mix of different elixir costs
            return self._select_balanced_elixir_troops(all_troops, parameters)
        elif template_type == "lane_adaptor":
            # Lane adaptor needs troops with different roles
            return self._select_role_balanced_troops(all_troops, parameters)
        elif template_type == "phase_based":
            # Phase-based needs a mix of cheap, medium, and expensive troops
            return self._select_cost_stratified_troops(all_troops, parameters)
        else:
            # Default to a random selection
            import random
            return random.sample(all_troops, 8)
    
    def _select_diverse_troops(self, all_troops: List[str], parameters: Dict[str, Any]) -> List[str]:
        """Select a diverse set of troops for counter picking."""
        # Ensure we have troops that can target air
        air_targeters = ["Archer", "Dragon", "Minion", "Wizard", "Musketeer"]
        
        # Ensure we have troops that can deal splash damage
        splash_troops = ["Dragon", "Wizard", "Valkyrie"]
        
        # Ensure we have some tanks
        tanks = ["Giant", "Prince", "Knight"]
        
        # Ensure we have some swarm troops
        swarms = ["Barbarian", "Skeleton", "Minion"]
        
        # Start with one from each category
        selected = [
            air_targeters[0],
            splash_troops[0],
            tanks[0],
            swarms[0]
        ]
        
        # Remove already selected troops from the categories
        air_targeters = [t for t in air_targeters if t not in selected]
        splash_troops = [t for t in splash_troops if t not in selected]
        tanks = [t for t in tanks if t not in selected]
        swarms = [t for t in swarms if t not in selected]
        
        # Add one more from each category if available
        if air_targeters:
            selected.append(air_targeters[0])
        if splash_troops:
            selected.append(splash_troops[0])
        if tanks:
            selected.append(tanks[0])
        if swarms:
            selected.append(swarms[0])
        
        # If we still need more, add from the remaining troops
        remaining = [t for t in all_troops if t not in selected]
        import random
        while len(selected) < 8 and remaining:
            troop = random.choice(remaining)
            selected.append(troop)
            remaining.remove(troop)
        
        return selected
    
    def _select_balanced_elixir_troops(self, all_troops: List[str], parameters: Dict[str, Any]) -> List[str]:
        """Select troops with a balanced distribution of elixir costs."""
        # Troop elixir costs
        troop_costs = {
            "Archer": 3,
            "Giant": 5,
            "Dragon": 4,
            "Balloon": 5,
            "Prince": 5,
            "Barbarian": 3,
            "Knight": 3,
            "Minion": 3,
            "Skeleton": 3,
            "Wizard": 5,
            "Valkyrie": 4,
            "Musketeer": 4
        }
        
        # Group troops by elixir cost
        cost_groups = {}
        for troop, cost in troop_costs.items():
            if cost not in cost_groups:
                cost_groups[cost] = []
            cost_groups[cost].append(troop)
        
        # Select troops with a balanced distribution of costs
        selected = []
        import random
        
        # Ensure we get at least 2 cheap troops (3 elixir)
        cheap_troops = cost_groups.get(3, [])
        if cheap_troops:
            selected.extend(random.sample(cheap_troops, min(3, len(cheap_troops))))
        
        # Ensure we get at least 2 medium troops (4 elixir)
        medium_troops = cost_groups.get(4, [])
        if medium_troops:
            selected.extend(random.sample(medium_troops, min(3, len(medium_troops))))
        
        # Ensure we get at least 2 expensive troops (5 elixir)
        expensive_troops = cost_groups.get(5, [])
        if expensive_troops:
            selected.extend(random.sample(expensive_troops, min(2, len(expensive_troops))))
        
        # If we still need more, add from the remaining troops
        remaining = [t for t in all_troops if t not in selected]
        while len(selected) < 8 and remaining:
            troop = random.choice(remaining)
            selected.append(troop)
            remaining.remove(troop)
        
        # If we have too many, trim the list
        if len(selected) > 8:
            selected = selected[:8]
        
        return selected
    
    def _select_role_balanced_troops(self, all_troops: List[str], parameters: Dict[str, Any]) -> List[str]:
        """Select troops with a balanced mix of roles for lane adaptation."""
        # Use roles from parameters if available, otherwise use defaults
        troop_roles = parameters.get("troop_roles", {
            "Giant": "tank",
            "Knight": "tank",
            "Prince": "tank",
            "Barbarian": "swarm",
            "Skeleton": "swarm",
            "Minion": "swarm",
            "Wizard": "splasher",
            "Valkyrie": "splasher",
            "Dragon": "splasher",
            "Archer": "support",
            "Musketeer": "support",
            "Balloon": "tank"
        })
        
        # Group troops by role
        role_groups = {}
        for troop, role in troop_roles.items():
            if role not in role_groups:
                role_groups[role] = []
            role_groups[role].append(troop)
        
        # Select troops with a balanced distribution of roles
        selected = []
        import random
        
        # Select 2 tanks
        tanks = role_groups.get("tank", [])
        if tanks:
            selected.extend(random.sample(tanks, min(2, len(tanks))))
        
        # Select 2 support troops
        supports = role_groups.get("support", [])
        if supports:
            selected.extend(random.sample(supports, min(2, len(supports))))
        
        # Select 2 splashers
        splashers = role_groups.get("splasher", [])
        if splashers:
            selected.extend(random.sample(splashers, min(2, len(splashers))))
        
        # Select 2 swarm troops
        swarms = role_groups.get("swarm", [])
        if swarms:
            selected.extend(random.sample(swarms, min(2, len(swarms))))
        
        # If we still need more, add from the remaining troops
        remaining = [t for t in all_troops if t not in selected]
        while len(selected) < 8 and remaining:
            troop = random.choice(remaining)
            selected.append(troop)
            remaining.remove(troop)
        
        # If we have too many, trim the list
        if len(selected) > 8:
            selected = selected[:8]
        
        return selected
    
    def _select_cost_stratified_troops(self, all_troops: List[str], parameters: Dict[str, Any]) -> List[str]:
        """Select troops stratified by cost category for phase-based strategies."""
        # Use cost categories from parameters if available, otherwise use defaults
        troop_costs = parameters.get("troop_costs", {
            "Archer": "cheap",
            "Barbarian": "cheap", 
            "Knight": "cheap",
            "Minion": "cheap",
            "Skeleton": "cheap",
            "Dragon": "medium",
            "Valkyrie": "medium",
            "Musketeer": "medium",
            "Giant": "expensive",
            "Prince": "expensive",
            "Wizard": "expensive",
            "Balloon": "expensive"
        })
        
        # Group troops by cost category
        cost_groups = {}
        for troop, cost in troop_costs.items():
            if cost not in cost_groups:
                cost_groups[cost] = []
            cost_groups[cost].append(troop)
        
        # Select troops with a stratified distribution of costs
        selected = []
        import random
        
        # Select 3 cheap troops
        cheap = cost_groups.get("cheap", [])
        if cheap:
            selected.extend(random.sample(cheap, min(3, len(cheap))))
        
        # Select 3 medium troops
        medium = cost_groups.get("medium", [])
        if medium:
            selected.extend(random.sample(medium, min(3, len(medium))))
        
        # Select 2 expensive troops
        expensive = cost_groups.get("expensive", [])
        if expensive:
            selected.extend(random.sample(expensive, min(2, len(expensive))))
        
        # If we still need more, add from the remaining troops
        remaining = [t for t in all_troops if t not in selected]
        while len(selected) < 8 and remaining:
            troop = random.choice(remaining)
            selected.append(troop)
            remaining.remove(troop)
        
        # If we have too many, trim the list
        if len(selected) > 8:
            selected = selected[:8]
        
        return selected