from flask import Flask, request
from flask_cors import CORS, cross_origin
from torch.utils.data import TensorDataset, DataLoader
from model import *
import mlflow.pytorch
import numpy as np
import torch
import math
import json
from preprocessor import *

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

model_dir = "../../models/modern/hw1pw1/MAE/models_e50/49/"

host = "10.112.79.143"
port = 5000
crt = "/etc/nginx/ssl/10.112.79.143.crt"
key = "/etc/nginx/ssl/10.112.79.143.key"

hw_unit = math.floor(HW * FPS / DOWNSAMPLE)
pw_unit = math.floor(PW * FPS / DOWNSAMPLE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        points = request.get_json()
        if not isinstance(points, (list)):
            raise ValueError("Request body is not a vaild array")
        points = np.array(points)
        data = torch.from_numpy(points)
        data = data.reshape(-1, 4).float()
        label = data.reshape(-1, 4).float()
        dataset = TensorDataset(data, label)
        datasetloader = DataLoader(dataset=dataset, batch_size=30)

        model = mlflow.pytorch.load_model(model_dir)
        model.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(datasetloader):
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(dim=0)
                pred = model(X)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                    pred = pred.tolist()
                return json.dumps(pred)


if __name__ == "__main__":
    app.run(debug=True, host=host, port=port, ssl_context=(crt, key))
