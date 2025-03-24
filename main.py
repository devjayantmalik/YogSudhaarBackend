import os
from typing import Tuple

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist

from src.predict import load_models, is_pose_correct

# define all paths
model_path = os.path.join("models", "models.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

# load all models
models = load_models(model_path, 0)
scaler = load_models(scaler_path, 1)


class PoseData(BaseModel):
    # Each pose will be tuple of (x, y, z, visibility)
    poses: conlist(Tuple[float, float, float, float], min_length=32, max_length=32)


class PoseFrames(BaseModel):
    """
    Example Input:
    { "frames": [
        { "poses" [ [x,y,z,visibility], [x,y,z,visibility ], ...(must be 32 items exact) },
        { "poses" [] },
        ...
    ]}
    """
    frames: conlist(PoseData, min_length=1, max_length=500)

    def to_pandas_df(self) -> pd.DataFrame:
        rows = []
        for frame in self.frames:
            row = []
            for pose in frame.poses:
                row.extend(pose)
            rows.append(row)

        return pd.DataFrame(rows)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Yog Sudhar. We hope you enjoy!"}


@app.post("/predictions/is-pose-correct/", status_code=200)
async def predict_is_pose_correct(data: PoseFrames):
    try:
        # Check if pose is correct for each frame
        is_correct = is_pose_correct(models, scaler, data.to_pandas_df())

        return {"is_pose_correct": is_correct,
                "message": "You are doing it correct." if is_correct else "You are doing it incorrect."}
    except Exception as ex:
        print(ex)
        return {"is_pose_correct": False,
                "message": "Failed to predict poses."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
