import sys
import os

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LSTGAN

def test_model_instantiation():
    """Validates that the LSTGAN model can be instantiated correctly."""
    model = LSTGAN()
    
    assert model.weekly_enc is not None
    assert model.daily_enc is not None
    assert model.local_enc is not None
    assert model.decoder is not None
    
    print("✅ Model instantiation successful.")
