import torch
import torch.nn as nn
from ultralytics.nn import tasks
from ghost_conv import GhostC2f
from attention import CBAM

def register_custom_modules():
    """Registers GhostC2f and CBAM into Ultralytics task parser for custom YAML support."""
    # Ensure custom modules are recognized by Ultralytics global class map
    tasks.GhostC2f = GhostC2f
    tasks.CBAM = CBAM
    
    # Patch parse_model if needed or append to task dictionary
    print("Successfully registered GhostC2f and CBAM modules into Ultralytics parser.")

if __name__ == "__main__":
    register_custom_modules()