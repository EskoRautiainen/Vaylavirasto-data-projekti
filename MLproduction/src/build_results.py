import pandas as pd

# ----------------------------------------------------------------------------------------------------
#   CATEGORIZATION
# ----------------------------------------------------------------------------------------------------
def categorize(scores):
    categories = []

    for s in scores: # Categorize anomaly scores
        if s <= -0.15:
            categories.append('Critical') 
        elif s <= -0.08:
            categories.append('Poor')
        elif s <= -0.03:
            categories.append('Fair')
        elif s <= 0.02:
            categories.append('Good')
        else:
            categories.append('Excellent')

    return categories


# ----------------------------------------------------------------------------------------------------
#   BUILD RESULTS
# ----------------------------------------------------------------------------------------------------
def step_06_build_results(metadata, features, predictions, scores):
    result = metadata.copy()

    result = result.join(features)

    result['anomaly_prediction'] = predictions
    result['anomaly_score'] = scores
    result['anomaly_type'] = [
        'Normal' if p == 1 else 'Anomaly' for p in predictions
    ]

    result['anomaly_category'] = categorize(scores)

    result['priority_score'] = result['anomaly_category'].map({ # Assign priority scores
        'Critical': 1,
        'Poor': 2,
        'Fair': 3,
        'Good': 4,
        'Excellent': 5
    })

    return result

