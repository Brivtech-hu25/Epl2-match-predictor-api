import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the trained model
model = joblib.load("epl_model_v2.joblib")

# Load historical matches
history = pd.read_csv("epl_history.csv")

history["Date"] = pd.to_datetime(
    history["Date"], dayfirst=True, errors="coerce"
)

history = history.dropna(
    subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
).sort_values("Date")

# Calculate each team's goals in historical matches
team_goals = {}

for _, match in history.iterrows():
    home = match["HomeTeam"]
    away = match["AwayTeam"]

    team_goals.setdefault(home, []).append(
        float(match["FTHG"])
    )
    team_goals.setdefault(away, []).append(
        float(match["FTAG"])
    )

app = FastAPI(title="EPL Match Predictor API")

# Temporary open CORS for frontend testing.
# We can restrict this to the website's domain later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    home_team: str
    away_team: str


@app.get("/")
def home():
    return {
        "message": "EPL Match Predictor API is running"
    }


@app.get("/teams")
def get_teams():
    return {"teams": sorted(team_goals.keys())}


@app.post("/predict")
def predict(request: MatchRequest):
    home = request.home_team.strip()
    away = request.away_team.strip()

    if home == away:
        raise HTTPException(
            status_code=400,
            detail="Choose two different teams."
        )

    if home not in team_goals or away not in team_goals:
        raise HTTPException(
            status_code=400,
            detail="One or both teams are not in the historical dataset."
        )

    home_last = team_goals[home][-5:]
    away_last = team_goals[away][-5:]

    home_avg = sum(home_last) / len(home_last)
    away_avg = sum(away_last) / len(away_last)

    match_input = pd.DataFrame([{
        "HomeTeam": home,
        "AwayTeam": away,
        "HomeAvgGoals5": home_avg,
        "AwayAvgGoals5": away_avg
    }])

    probabilities = model.predict_proba(match_input)[0]

    result = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}

    for label, probability in zip(model.classes_, probabilities):
        if label == "H":
            result["home_win"] = round(float(probability) * 100, 2)
        elif label == "D":
            result["draw"] = round(float(probability) * 100, 2)
        elif label == "A":
            result["away_win"] = round(float(probability) * 100, 2)

    return {
        "home_team": home,
        "away_team": away,
        "probabilities": result,
        "historical_data_season": "2023-24"
    }
