import json
import os
import time

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
    poses: list[conlist(float, min_length=133, max_length=133)]


class PoseFrames(BaseModel):
    """
    Example Input:
    { "frames": [
        { "poses" [ frame, x_0, y_0, z_0, visibility_0, x_1, y_1, z_1, visibility_1 , ...(must be 133 items exact) ] },
        { "poses" [] },
        ...
    ]}
    """
    frames: conlist(PoseData, min_length=1, max_length=500)

    def to_pandas_df(self) -> pd.DataFrame:
        columns = ["frame"]
        for i in range(33):
            columns.append(f"x_{i}")
            columns.append(f"y_{i}")
            columns.append(f"z_{i}")
            columns.append(f"visibility_{i}")

        rows = [pose for frame in self.frames for pose in frame.poses]
        return pd.DataFrame(rows, columns=columns)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Yog Sudhar. We hope you enjoy!"}


@app.post("/predictions/is-pose-correct", status_code=200)
async def predict_is_pose_correct(data: PoseFrames):
    try:
        # Check if pose is correct for each frame
        print(f"Predicting for data: {json.dumps(data)}")
        df = data.to_pandas_df()
        basename = f"requests/{time.strftime('%m-%d-%Y--%H-%M-%S')}"

        # enable for logging and debugging
        df.to_csv(basename + ".request.csv")

        # get prediction
        is_correct = is_pose_correct(models, scaler, df)
        response = {"is_pose_correct": is_correct,
                    "message": "You are doing it correct." if is_correct else "You are doing it incorrect."}

        # save for logging and debugging
        with open(basename + ".success.txt", "w") as file:
            file.write(json.dumps(response))

        return response
    except Exception as ex:
        # save for logging and debugging
        with open(basename + ".exception.txt", "w") as file:
            file.write(str(ex))

        return {"is_pose_correct": False,
                "message": "Failed to predict poses."}


if __name__ == "__main__":
    os.makedirs("requests", exist_ok=True) # for logging.
    uvicorn.run(app, host="0.0.0.0", port=8009, root_path="/yog-sudhar")
