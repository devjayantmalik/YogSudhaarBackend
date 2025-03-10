import os

import uvicorn
from fastapi import FastAPI

from src.predict import load_models

# define all paths
model_path = os.path.join("models", "models.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

# load all models
models = load_models(model_path, 0)
scaler = load_models(scaler_path, 1)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Yog Sudhar. We hope you enjoy!"}


@app.post("/predictions/is-pose-correct/")
async def predict_is_pose_correct():
    return {"is_pose_correct": True, "message": "You are doing it correct."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
