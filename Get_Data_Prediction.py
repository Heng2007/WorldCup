"""Feature engineering for a single upcoming fixture.

Loads the historical and current-cycle match data at import time, then exposes
``GetData``, which turns a (home team, away team) pair into the one-row feature
frame that the trained model expects.
"""

import requests
import json
import pandas as pd
import ast
from Get_Data_Training import Model

df = pd.read_csv("matches_1930_2022.csv")
cleaned_df = df.dropna(how ="all")
#-------------------------------------------------------------------------------------------------
df1 = pd.read_csv("2026WorldCupLatest.csv")
df1 = df1.drop(columns = ["utcDate","id","area","season","competition","stage","group","referees","odds"])
cleaned_df1 = df1.dropna(how="all")
#-------------------------------------------------------------------------------------------------
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


#-------------------------------------------------------------------------------------------------


def convert(raw: str):
    return ast.literal_eval(raw)

def show_everything():
    pd.set_option("display.max_columns", None)
show_everything()

class GetData:
    """Builds the six-feature row that the trained model expects for one fixture.

    The scoring averages blend two sources 50/50: all-time historical form from
    ``results.csv`` and current-cycle form from the football-data.org export in
    ``2026WorldCupLatest.csv``. This keeps a team that has changed a lot recently
    from being judged purely on its all-time record.

    Team names must match the source data exactly, e.g. ``"Canada"``.

    Example:
        >>> GetData("Canada", "Switzerland").get_data()
    """

    def __init__(self, home_team:str, away_team:str):
        self.home_team = home_team
        self.away_team = away_team
#--------------------------Get the historical head to head result of two teams----------------------------
    def face_to_face(self) :
        """Return the historical head-to-head record between the two teams.

        Matches where the roles are reversed are flipped so every row is read
        from ``self.home_team``'s perspective before tallying.

        Returns:
            tuple[str, int]: the dominant team's name and how many more matches
            it won, or ``("draw", 0)`` if the record is level.
        """
        records = cleaned_df[(cleaned_df["home_team"]==self.home_team)
                             &(cleaned_df["away_team"]==self.away_team)].copy()

        records1 = cleaned_df[(cleaned_df["away_team"] == self.home_team)
                                & (cleaned_df["home_team"] == self.away_team)].copy()
        records1[["home_team","away_team"]] = records1[["away_team","home_team"]]
        records1[["home_score","away_score"]] = records1[["away_score","home_score"]]
        records = pd.concat([records1, records])
        home_index = 0
        for index, row in records.iterrows():
            if row["home_score"] < row["away_score"]:
                home_index -=1
            elif row["home_score"] > row["away_score"]:
                home_index +=1

        if home_index < 0:
            return self.away_team, -home_index
        elif home_index > 0:
            return self.home_team, home_index
        else:
            return "draw", home_index
#----------------------home_average_goal blend historical data and current data with a ratio of 5/5------------------
    def home_average_goal(self):
        """Average goals scored by the home team, blended 50/50 current vs all-time.

        Returns 0 for the current-cycle half if the team has no played matches
        yet in the current data (all scores still ``None``).
        """
        list1 = []
        serie = df1[df1["homeTeam"].apply(lambda x: convert(x)["name"] == self.home_team)]
        for value in serie["score"]:
            value = convert(value)
            list1.append((value["fullTime"]["home"], value["fullTime"]["away"]))
        serie1 = df1[df1["awayTeam"].apply(lambda x: convert(x)["name"] == self.home_team)]
        for value in serie1["score"]:
            value = convert(value)
            list1.append((value["fullTime"]["away"], value["fullTime"]["home"]))
        try:
            list1.remove((None, None))
        except ValueError:
            pass
        # current_record = sum(x[0] for x in list1) / len(list1)
        filtered = [x for x in list1 if x[0] is not None]
        if len(filtered) == 0:
            current_record = 0
        else:
            current_record = sum(x[0] for x in filtered) / len(filtered)

        historical_record = data[data["home_team"].apply(lambda x: x == self.home_team)]["home_avg_goals"].iloc[0]
        overall_record = current_record*0.5 + historical_record*0.5
        return overall_record
    #---------------home_average_goal blend historical data and current data with a ratio of 5/5---------------------
    def away_average_goal(self):
        """Average goals scored by the away team.

        Reuses :meth:`home_average_goal` by swapping the two teams around.
        """
        away_team = GetData(self.away_team, self.home_team)
        return away_team.home_average_goal()
    #--------------home_avg_conceded--------------------------------------------------------------------------
    def home_avg_conceded(self):
        """Average goals conceded by the home team, blended 50/50 current vs all-time."""
        list1 = []
        serie = df1[df1["homeTeam"].apply(lambda x: convert(x)["name"] == self.home_team)]
        for value in serie["score"]:
            value = convert(value)
            list1.append((value["fullTime"]["home"], value["fullTime"]["away"]))
        serie1 = df1[df1["awayTeam"].apply(lambda x: convert(x)["name"] == self.home_team)]
        for value in serie1["score"]:
            value = convert(value)
            list1.append((value["fullTime"]["away"], value["fullTime"]["home"]))
        try:
            list1.remove((None, None))
        except ValueError:
            pass
        # current_record = sum(x[1] for x in list1) / len(list1)
        filtered = [x for x in list1 if x[1] is not None]
        if len(filtered) == 0:
            current_record = 0
        else:
            current_record = sum(x[1] for x in filtered) / len(filtered)

        historical_record = data[data["home_team"].apply(lambda x: x == self.home_team)]["away_avg_goals"].iloc[0]
        overall_record = current_record*0.5 + historical_record*0.5
        return overall_record
    #---------------away_avg_conceded________________________________________________________________
    def away_avg_conceded(self):
        """Average goals conceded by the away team.

        Reuses :meth:`home_avg_conceded` by swapping the two teams around.
        """
        away_team = GetData(self.away_team, self.home_team)
        return away_team.home_avg_conceded()
    #--------------head2head_adv---------------------------------------------------------------------
    def head2head_adv(self):
        """Home team's historical win rate in this exact fixture.

        Falls back to 0.5 (no advantage either way) when the two teams have
        never met in the training data.

        Note:
            This feature leaks — see "Known limitations" in the README. It is
            computed across the whole dataset, so a training row's own result
            contributes to its own feature value.
        """
        try: number = data[(data["home_team"] == self.home_team)&(data["away_team"] == self.away_team)]["head2head_adv"].iloc[0]
        except IndexError:
            return 0.5
        return number
    #------------get_data--------------------------------------------------------------
    def get_data(self):
        """Assemble the single-row DataFrame consumed by ``Model.predict``.

        Column order and names must match the training features in
        ``Get_Data_Training.py`` exactly, or scikit-learn will reject the input.

        Returns:
            pd.DataFrame: one row, six feature columns.
        """
        match = pd.DataFrame([{
            "home_avg_goals":self.home_average_goal(),
            "away_avg_goals":self.away_average_goal(),
            "home_avg_conceded":self.home_avg_conceded(),
            "away_avg_conceded":self.away_avg_conceded(),
            "head2head_adv":self.head2head_adv(),
            "neutral_index": 0
        }])
        return match


# object2 = GetData(home_team="Canada", away_team="Switzerland")
# match = pd.DataFrame([{
#     "home_avg_goals":object2.home_average_goal(),
#     "away_avg_goals":object2.away_average_goal(),
#     "home_avg_conceded":object2.home_avg_conceded(),
#     "away_avg_conceded":object2.away_avg_conceded(),
#     "head2head_adv":object2.head2head_adv(),
#     "neutral_index": 0
# }])
# mode = Model(match)
# print(f"The prediction of the game is: {mode.predict()}")


# print(match)
