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
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("results.csv")
data = data.dropna(axis = 0, how = "any")

#Adding the result column
data["result"] = "Draw"
data.loc[data["home_score"] > data["away_score"], ["result"]] = "Win"
data.loc[data["home_score"] < data["away_score"], ["result"]] = "Loss"

#Adding the home_avg_goals
data["home_avg_goals"] = data.groupby("home_team")["home_score"].transform("mean")
#Adding the away_avg_goals
data["away_avg_goals"] = data.groupby("away_team")["away_score"].transform("mean")
#Adding the home_avg_conceded
data["home_avg_conceded"] = data.groupby("home_team")["away_score"].transform("mean")
#Adding the away_avg_conceded
data["away_avg_conceded"] = data.groupby("away_team")["home_score"].transform("mean")
#Adding the head2head_adv
number_per_group = data.groupby(["home_team","away_team"])["home_team"].transform("count")
win_per_group = data.groupby(["home_team","away_team"])["result"].transform(lambda x: (x == "Win").sum())
data["head2head_adv"] = win_per_group/number_per_group
#Adding the neutral_index 0 if neutral 1 for not neutral
data["neutral_index"] = data["neutral"].astype(int)



Y = data["result"]
X = data[["home_avg_goals", "away_avg_goals", "home_avg_conceded", "away_avg_conceded",
            "head2head_adv", "neutral_index"]]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,
                                                    random_state = 0)

model = LogisticRegression(C = 100, class_weight="balanced", solver="lbfgs",
                           max_iter=1000, verbose=0)

model.fit(x_train, y_train)


# print("actual result:")
# print(data.iloc[101]["result"])

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