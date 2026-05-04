"""Model Evaluation - Evaluate severity and risk score models"""

# TODO: Implement evaluate_all_models() to evaluate both trained models on held-out evaluation datasets
# PURPOSE: Load pre-trained severity classifier and risk score regressor, evaluate on evaluation CSVs, compute metrics (accuracy, precision, recall, F1 for classifier; MAE, RMSE, R2 for regressor)
# PARAMETERS: Optional evaluation_dir path (default "data/evaluation_dataset"), optional model_dir path (default "ml/models")
# RETURNS: Dictionary with status, models evaluation results with metrics, handles missing models/files gracefully with non-blocking error handling

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def evaluate_all_models(evaluation_dir="data/evaluation_dataset", model_dir="ml/models"):
    """
    Evaluate both trained models on held-out evaluation datasets.
    
    Parameters:
        evaluation_dir: Path to evaluation dataset directory
        model_dir: Path to trained models directory
    
    Returns:
        Dictionary with evaluation status and metrics
    """
    results = {
        "status": "success",
        "models": {
            "severity_classifier": {"metrics": {}},
            "risk_score_regressor": {"metrics": {}}
        }
    }
    
    try:
        # Evaluate Severity Model
        try:
            severity_model = joblib.load(f"{model_dir}/severity_model.pkl")
            severity_scaler = joblib.load(f"{model_dir}/severity_scaler.pkl")
            
            severity_data = pd.read_csv(f"{evaluation_dir}/severity_eval.csv")
            
            feature_cols = [
                'patient_age_years', 'symptom_duration_hours', 'fever_present', 'neck_stiffness',
                'body_temperature_celsius', 'heart_rate_bpm', 'blood_pressure_systolic_mmhg',
                'blood_pressure_diastolic_mmhg', 'respiratory_rate_breaths_per_minute',
                'oxygen_saturation_percent', 'comorbidities_count'
            ]
            
            X_severity = severity_data[feature_cols]
            y_severity = severity_data['severity_level']
            
            # Scale and predict (NO label encoder needed!)
            X_severity_scaled = severity_scaler.transform(X_severity)
            y_pred = severity_model.predict(X_severity_scaled)
            
            # Calculate metrics directly
            accuracy = accuracy_score(y_severity, y_pred)
            precision = precision_score(y_severity, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_severity, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_severity, y_pred, average='weighted', zero_division=0)
            
            results["models"]["severity_classifier"]["metrics"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "sample_count": len(X_severity),
                "feature_count": len(feature_cols)
            }
        except Exception as e:
            results["models"]["severity_classifier"]["metrics"] = {"error": str(e)}
        
        # Evaluate Risk Score Model
        try:
            risk_model = joblib.load(f"{model_dir}/risk_score_model.pkl")
            risk_scaler = joblib.load(f"{model_dir}/risk_score_scaler.pkl")
            
            risk_data = pd.read_csv(f"{evaluation_dir}/risk_score_eval.csv")
            
            feature_cols = [
                'patient_age_years', 'symptom_duration_hours', 'fever_present', 'neck_stiffness',
                'body_temperature_celsius', 'heart_rate_bpm', 'blood_pressure_systolic_mmhg',
                'blood_pressure_diastolic_mmhg', 'respiratory_rate_breaths_per_minute',
                'oxygen_saturation_percent', 'comorbidities_count'
            ]
            
            X_risk = risk_data[feature_cols]
            y_risk = risk_data['risk_score']
            
            X_risk_scaled = risk_scaler.transform(X_risk)
            y_pred_risk = risk_model.predict(X_risk_scaled)
            y_pred_risk = np.clip(y_pred_risk, 0.0, 1.0)
            
            mae = mean_absolute_error(y_risk, y_pred_risk)
            rmse = np.sqrt(mean_squared_error(y_risk, y_pred_risk))
            r2 = r2_score(y_risk, y_pred_risk)
            
            results["models"]["risk_score_regressor"]["metrics"] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2_score": float(r2),
                "sample_count": len(X_risk),
                "feature_count": len(feature_cols)
            }
        except Exception as e:
            results["models"]["risk_score_regressor"]["metrics"] = {"error": str(e)}
    
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


if __name__ == "__main__":
    import json
    results = evaluate_all_models()
    print(json.dumps(results, indent=2))
