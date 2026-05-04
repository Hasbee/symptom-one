"""ML Training Pipeline - Orchestrate model training"""

# TODO: Implement run_training_pipeline() to orchestrate complete ML model training workflow
# PURPOSE: Execute severity classifier and risk score regressor training sequentially, then evaluate both models
# PARAMETERS: None (uses hardcoded paths for training data, models directory)
# RETURNS: Boolean indicating success (True if both models trained successfully, False otherwise)

from .train_model.train_severity_model import train_severity_model
from .train_model.train_risk_score_model import train_risk_score_model


def run_training_pipeline():
    """Execute complete ML model training workflow"""
    print("\n" + "="*60)
    print("Starting ML Training Pipeline")
    print("="*60 + "\n")
    
    # Train severity model
    print("1. Training Severity Assessment Model...")
    severity_success = train_severity_model()
    
    print()
    
    # Train risk score model
    print("2. Training Risk Score Model...")
    risk_success = train_risk_score_model()
    
    print("\n" + "="*60)
    if severity_success and risk_success:
        print("✓ Training pipeline completed successfully!")
        print("="*60 + "\n")
        return True
    else:
        print("✗ Training pipeline failed!")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    run_training_pipeline()
