"""Trains the Win/Draw/Loss classifier on historical international results.

Engineers six features from ``results.csv``, fits a logistic regression, and
exposes ``Model`` as a thin wrapper for making predictions.

Note:
    Training runs at import time, so importing this module refits the model from
    scratch. Persisting the fitted estimator with ``joblib`` is on the roadmap.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("results.csv")
data = data.dropna(axis = 0, how = "any")



#Adding the result column
data["result"] = "Draw"
data.loc[data["home_score"] > data["away_score"], ["result"]] = "Win"
data.loc[data["home_score"] < data["away_score"], ["result"]] = "Loss"
#Adding the neutral_index 0 if neutral 1 for not neutral
data["neutral_index"] = data["neutral"].astype(int)


train_rows, test_rows = train_test_split(data, test_size = 0.2, random_state = 0)

#Lookup tables, built from the training rows only
home_goals_map = train_rows.groupby("home_team")["home_score"].mean()
away_goals_map = train_rows.groupby("away_team")["away_score"].mean()
home_conceded_map = train_rows.groupby("home_team")["away_score"].mean()
away_conceded_map = train_rows.groupby("away_team")["home_score"].mean()
head2head_map = train_rows.groupby(["home_team","away_team"])["result"].apply(
    lambda x: (x == "Win").mean())

# Build an average case as a fallback
DEFAULTS = {"home_avg_goals": train_rows["home_score"].mean(),
            "away_avg_goals": train_rows["away_score"].mean(),
            "home_avg_conceded": train_rows["away_score"].mean(),
            "away_avg_conceded": train_rows["home_score"].mean(),
            "head2head_adv": 0.5}


def build_features(rows):
    """Build the six feature columns for ``rows`` from the training-only tables.
    """
    features = pd.DataFrame(index = rows.index)
    features["home_avg_goals"] = rows["home_team"].map(home_goals_map)
    features["away_avg_goals"] = rows["away_team"].map(away_goals_map)
    features["home_avg_conceded"] = rows["home_team"].map(home_conceded_map)
    features["away_avg_conceded"] = rows["away_team"].map(away_conceded_map)
    pairs = pd.MultiIndex.from_arrays([rows["home_team"], rows["away_team"]])
    features["head2head_adv"] = head2head_map.reindex(pairs).to_numpy()
    features["neutral_index"] = rows["neutral_index"].to_numpy()
    return features.fillna(DEFAULTS)


x_train = build_features(train_rows)
x_test = build_features(test_rows)
y_train = train_rows["result"]
y_test = test_rows["result"]



model = LogisticRegression(C = 100, class_weight="balanced", solver="lbfgs",
                           max_iter=1000, verbose=0)

model.fit(x_train, y_train)




class Model:
    """Wraps the fitted classifier so callers pass features once and predict.

    Args:
        features: a DataFrame whose columns match the training features above,
            typically produced by ``GetData.get_data()``.

    Example:
        >>> Model(GetData("Canada", "Switzerland").get_data()).predict()
        array(['Loss'], dtype=object)
    """

    def __init__(self,features):
        self.features = features

    def predict(self):
        """Return the predicted outcome(s) as a numpy array of label strings."""
        model_prediction = model.predict(self.features)
        return model_prediction

#----------------------------------------------------------------------------------------------------
# # Build one row of features
# match = pd.DataFrame([{
#     "home_avg_goals":    1.894231,
#     "away_avg_goals":    0.847826,
#     "home_avg_conceded":  1.012019,
#     "away_avg_conceded": 2.032609,
#     "head2head_adv":     0.679245,
#     "neutral_index": 0
# }])
#
#
# # Predict
# prediction = model.predict(match)
#
# print("predicted result:")
# print(prediction[0])