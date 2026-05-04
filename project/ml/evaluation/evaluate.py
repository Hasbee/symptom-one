"""ML Evaluation - Run model evaluation"""

import json

from ml.evaluation import evaluate_all_models

if __name__ == "__main__":
    results = evaluate_all_models()
    print(json.dumps(results, indent=2))
