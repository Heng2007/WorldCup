"""Command-line entry point: prompts for two teams and prints the prediction.

Note:
    The prompts run at import time. Wrapping them in an
    ``if __name__ == "__main__":`` guard is on the roadmap.
"""

from Get_Data_Training import Model
from Get_Data_Prediction import GetData
import Get_Data_Training
from sklearn.metrics import classification_report
# home_team = input("Enter home team name: ")
# away_team = input("Enter away team name: ")
# data = GetData(home_team,away_team).get_data()
# prediction = Model(data)
# result = str(prediction.predict()).lstrip("['").rstrip("']")

# print(f"{home_team} will {result}")

s = Model(Get_Data_Training.x_test)
x = classification_report(Get_Data_Training.y_test, s.predict())
print(x)