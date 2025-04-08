import pickle
from typing import Any

import numpy as np
import pandas

from . import normalize, params, pre_processing


# import normalize
# from . import params
# import pre_processing


# Always ensure that you have "model.pkl in Train_model folder". If not then first run geenerate_model.py in Train_model.
def load_models(model_path, x):
    try:
        with open(model_path, "rb") as file:
            models = pickle.load(file)
        if x == 0:
            print("Models loaded successfully.")
        else:
            print("Scaler loaded successfully.")
        return models
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        raise
    except pickle.UnpicklingError as e:
        print(f"Error unpickling the models: {e}")
        raise


def predict_all(models, input_data):
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(input_data)
    return predictions


def is_pose_correct(models: Any, scaler: Any, data: pandas.DataFrame) -> bool:
    # Normalise csv data
    obj = normalize.LandmarksDataset()
    df = obj.process_data_from_excel(data)

    # Pre-processing on the data.
    X_test = pre_processing.single_processor(df)

    # load the min-max scaler for preprocessing
    X_test = scaler.transform(X_test)

    # get predictions from all models
    predictions = predict_all(models, X_test)

    print("\nPredictions for each model:")
    target_nm = ['cor']
    target_nm.extend(params.inc_labels)
    counter = np.zeros(len(target_nm))
    for model_name, model_predictions in predictions.items():
        counter[model_predictions] += 1

    prediction = target_nm[np.argmax(counter)]
    return prediction == "cor" if (prediction == "cor") else prediction

# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     csv_path = os.path.join(current_dir, 'test_csv')
#
#     model_path = "Train_model/model.pkl"
#     scaler_path = "Train_model/scaler.pkl"
#
#     try:
#         data = pd.read_csv(csv_path, header=0, index_col=None)
#         print("CSV file loaded successfully.")
#
#         # obj = normalize_csv.LandmarksDataset()
#         # df = obj.process_data_from_excel(data)
#         # print("Normalization completed.")
#         # print(df)
#
#         # Pre-processing on the data.
#         # X_test = pre_processing.single_processor(df)
#         # print(X_test)
#
#         # load the min-max scaler for preprocessing
#         # scaler = load_models(scaler_path, 1)
#         # X_test = scaler.transform(X_test)
#
#         # models = load_models(model_path, 0)
#         # predictions = []
#         # predictions = predict_all(models, X_test)
#
#         # print("\nPredictions for each model:")
#         # target_nm = ['cor']
#         # target_nm.extend(params.inc_labels)
#         # counter = np.zeros(len(target_nm))
#         # for model_name, model_predictions in predictions.items():
#         #     print(f"Model: {model_name}, Prediction: {model_predictions}")
#         #     counter[model_predictions] += 1
#
#         # print(counter)
#
#         # print(f'\nFinal Class-Prediction: {target_nm[np.argmax(counter)]}')
#
#         # Save predictions to a file (optional)
#         # output_predictions_csv = "predictions.csv"
#         # pd.DataFrame(predictions).to_csv(output_predictions_csv, index=False)
#         # print(f"Predictions saved to {output_predictions_csv}.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
