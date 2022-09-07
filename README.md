# Master-Thesis-Gesture-Recognition-Model_Training
Python script for LSTM hyperparameter tuning and model training. 

# Dependencies
- Python 3.9.2
- matplotlib 3.5.1
- scipy 1.7.3
- scikit-learn  1.0.2
- numpy 1.23.0
- tensorflow 2.7.0
- keras-tuner 1.1.3
# Data
Training, validation and testing data and their labels are included in the "Data" folder. Each sample is of shape (14,33) where each row represents the normalized fast-point-feature histogram of a point cloud frame.
# Pretrained Models
Pretrained models on batch sizes 16,32,48 and 64 are available. They can be accessed via `tensorflow.keras.models.load_model("model_name")` and `model.summary()`

