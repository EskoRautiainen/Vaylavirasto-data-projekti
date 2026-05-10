import pandas as pd

# ----------------------------------------------------------------------------------------------------
#   CATEGORIZATION
# ----------------------------------------------------------------------------------------------------
def categorize(scores):
    s = pd.Series(scores)
    categories = []
    percentile = s.rank(pct=True)

    return pd.cut(
        percentile,
        bins=[0, 0.04, 0.08, 0.4, 0.8, 1.0],            
        labels=["Critical", "Poor", "Fair", "Good", "Excellent"]
    )


# ----------------------------------------------------------------------------------------------------
#   BUILD RESULTS
# ----------------------------------------------------------------------------------------------------
def step_06_build_results(metadata, features, predictions, scores):
    result = metadata.copy()

    result = result.join(features)

    result['anomaly_prediction'] = predictions
    result['anomaly_score'] = scores.round(1)
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

