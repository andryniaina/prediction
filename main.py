from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()

# Mock data for training
class VoteResult(BaseModel):
    date: str

# Updated data for today and the next 10 days
data = [
    {"date": "2024-09-23", "results": {"66efe17cd2944247b3830f62": 40, "66efe325d2944247b3830f72": 20, "66efe35bd2944247b3830f75": 20, "66efe2e4d2944247b3830f6f": 20, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-09-24", "results": {"66efe17cd2944247b3830f62": 42, "66efe325d2944247b3830f72": 18, "66efe35bd2944247b3830f75": 19, "66efe2e4d2944247b3830f6f": 21, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-09-25", "results": {"66efe17cd2944247b3830f62": 43, "66efe325d2944247b3830f72": 19, "66efe35bd2944247b3830f75": 18, "66efe2e4d2944247b3830f6f": 20, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-09-26", "results": {"66efe17cd2944247b3830f62": 41, "66efe325d2944247b3830f72": 21, "66efe35bd2944247b3830f75": 19, "66efe2e4d2944247b3830f6f": 19, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-09-27", "results": {"66efe17cd2944247b3830f62": 44, "66efe325d2944247b3830f72": 19, "66efe35bd2944247b3830f75": 20, "66efe2e4d2944247b3830f6f": 17, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-09-28", "results": {"66efe17cd2944247b3830f62": 45, "66efe325d2944247b3830f72": 17, "66efe35bd2944247b3830f75": 18, "66efe2e4d2944247b3830f6f": 20, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-09-29", "results": {"66efe17cd2944247b3830f62": 46, "66efe325d2944247b3830f72": 16, "66efe35bd2944247b3830f75": 19, "66efe2e4d2944247b3830f6f": 19, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-09-30", "results": {"66efe17cd2944247b3830f62": 48, "66efe325d2944247b3830f72": 14, "66efe35bd2944247b3830f75": 18, "66efe2e4d2944247b3830f6f": 20, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-10-01", "results": {"66efe17cd2944247b3830f62": 47, "66efe325d2944247b3830f72": 15, "66efe35bd2944247b3830f75": 18, "66efe2e4d2944247b3830f6f": 20, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-10-02", "results": {"66efe17cd2944247b3830f62": 49, "66efe325d2944247b3830f72": 12, "66efe35bd2944247b3830f75": 19, "66efe2e4d2944247b3830f6f": 20, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}},
    {"date": "2024-10-03", "results": {"66efe17cd2944247b3830f62": 50, "66efe325d2944247b3830f72": 13, "66efe35bd2944247b3830f75": 17, "66efe2e4d2944247b3830f6f": 20, "66efe21ed2944247b3830f69": 0, "66efe29dd2944247b3830f6c": 0}}
]

model = None

# Prepare the data for TensorFlow model
def prepare_data():
    inputs = []
    outputs = []
    for entry in data:
        inputs.append(list(entry["results"].values()))  # Inputs: list of the 6 vote results
        outputs.append(list(entry["results"].values())) # Outputs: list of the same 6 results (as we want to predict the 6 keys)
    return np.array(inputs), np.array(outputs)

# Create a TensorFlow model to predict 6 vote results
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # 6 static vote keys
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6)  # Output 6 values for the 6 static keys
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.post("/train")
async def train_model():
    global model
    x_train, y_train = prepare_data()
    model = create_model()
    model.fit(x_train, y_train, epochs=100)
    return {"message": "Model trained successfully"}

@app.post("/predict")
async def predict(vote_result: VoteResult):
    if model is None:
        return {"error": "Model is not trained yet"}
    
    # Generate a dummy input for the model, using the average of past results
    input_data = np.mean(np.array([list(entry["results"].values()) for entry in data]), axis=0).reshape(1, 6)  # Average input data
    
    prediction = model.predict(input_data)[0]  # Get the prediction for all 6 vote keys
    
    # Normalize the prediction so that the total sums to 100
    total = np.sum(prediction)
    normalized_prediction = (prediction / total) * 100
    
    # Convert the normalized predictions to integers
    int_predictions = np.floor(normalized_prediction).astype(int)
    sum_int_predictions = np.sum(int_predictions)
    
    # Adjust the integers to ensure the sum is exactly 100
    difference = 100 - sum_int_predictions
    if difference > 0:
        # Add 1 to the elements with the highest decimal points until the total is 100
        fractional_parts = normalized_prediction - int_predictions
        adjustment_indices = np.argsort(fractional_parts)[-difference:]  # Indices with highest fractional parts
        for i in adjustment_indices:
            int_predictions[i] += 1

    # Create the result dictionary with static keys
    keys = ["66efe17cd2944247b3830f62", "66efe21ed2944247b3830f69", "66efe29dd2944247b3830f6c", "66efe2e4d2944247b3830f6f", "66efe325d2944247b3830f72", "66efe35bd2944247b3830f75"]
    
    result = {key: int(pred) for key, pred in zip(keys, int_predictions)}
    return result

@app.get("/")
async def root():
    return {"message": "Vote result prediction API is running"}