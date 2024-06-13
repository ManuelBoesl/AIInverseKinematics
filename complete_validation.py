from direct_kin import get_trans_and_rot
import matplotlib.pyplot as plt
from keras.api import models
import numpy as np
import joblib
from custom_loss import custom_loss
from custom_loss import custom_loss_signle_mapping
from plot_data import plot_3d
from regression import regression
import plotly.express as px

# load the pre-trained model
model = models.load_model('my_model.keras', custom_objects={'custom_loss': custom_loss})

input_data = np.load('input_data.npy')
target_data = np.load('target_data.npy')

# take 20 percent of the data for testing
test_size = int(input_data.shape[0] * 0.2)
test_input_data = input_data[:test_size]
test_target_data = target_data[:test_size]

position_errors_nn = []
orientation_errors_nn = []

position_errors_regression = []
orientation_errors_regression = []
iteration_numbers = []

for i in range(test_size):
    # get the predicted thetas
    predicted_thetas = model.predict(test_input_data[i].reshape(1, 6))
    # get the predicted position and orientation
    trans_and_rot_predicted = get_trans_and_rot(predicted_thetas[0]).reshape(6, 1)

    # get the original position and orientation
    trans_and_rot_original = test_target_data[i].reshape(6, 1)

    # calculate the error
    position_error_nn = np.sqrt(np.sum((trans_and_rot_original[:3] - trans_and_rot_predicted[:3]) ** 2))
    orientation_error_nn = np.sqrt(np.sum((trans_and_rot_original[3:] - trans_and_rot_predicted[3:]) ** 2))

    position_errors_nn.append(position_error_nn)
    orientation_errors_nn.append(orientation_error_nn)

    # regression
    regression_theta, iteration = regression(test_input_data[i].reshape(6, 1), predicted_thetas, learning_rate=0.5)
    trans_and_rot_regression = get_trans_and_rot(regression_theta).reshape(6, 1)

    position_error_regression = np.sqrt(np.sum((trans_and_rot_original[:3] - trans_and_rot_regression[:3]) ** 2))
    orientation_error_regression = np.sqrt(np.sum((trans_and_rot_original[3:] - trans_and_rot_regression[3:]) ** 2))

    position_errors_regression.append(position_error_regression)
    orientation_errors_regression.append(orientation_error_regression)
    iteration_numbers.append(iteration)

    print(f"Position Error NN: {position_error_nn}; Orientation Error NN: {orientation_error_nn}")
    print(f"Position Error Regression: {position_error_regression}; Orientation Error Regression: {orientation_error_regression}")
    print(f"Iteration: {iteration}")

# calculate the mean error, mean iteration and the standard deviation
position_errors_nn = np.array(position_errors_nn)
orientation_errors_nn = np.array(orientation_errors_nn)
position_errors_regression = np.array(position_errors_regression)
orientation_errors_regression = np.array(orientation_errors_regression)
iteration_numbers = np.array(iteration_numbers)

mean_position_error_nn = np.mean(position_errors_nn)
mean_orientation_error_nn = np.mean(orientation_errors_nn)
mean_position_error_regression = np.mean(position_errors_regression)
mean_orientation_error_regression = np.mean(orientation_errors_regression)
mean_iteration = np.mean(iteration_numbers)

std_position_error_nn = np.std(position_errors_nn)
std_orientation_error_nn = np.std(orientation_errors_nn)
std_position_error_regression = np.std(position_errors_regression)
std_orientation_error_regression = np.std(orientation_errors_regression)
std_iteration = np.std(iteration_numbers)

max_position_error_nn = np.max(position_errors_nn)
max_orientation_error_nn = np.max(orientation_errors_nn)
max_position_error_regression = np.max(position_errors_regression)
max_orientation_error_regression = np.max(orientation_errors_regression)
max_iteration = np.max(iteration_numbers)

min_position_error_nn = np.min(position_errors_nn)
min_orientation_error_nn = np.min(orientation_errors_nn)
min_position_error_regression = np.min(position_errors_regression)
min_orientation_error_regression = np.min(orientation_errors_regression)
min_iteration = np.min(iteration_numbers)

print(f"Mean Position Error NN: {mean_position_error_nn}; Mean Orientation Error NN: {mean_orientation_error_nn}")
print(f"Mean Position Error Regression: {mean_position_error_regression}; Mean Orientation Error Regression: {mean_orientation_error_regression}")
print(f"Mean Iteration: {mean_iteration}")
print(f"Standard Deviation Position Error NN: {std_position_error_nn}; Standard Deviation Orientation Error NN: {std_orientation_error_nn}")
print(f"Standard Deviation Position Error Regression: {std_position_error_regression}; Standard Deviation Orientation Error Regression: {std_orientation_error_regression}")
print(f"Standard Deviation Iteration: {std_iteration}")
print(f"Max Position Error NN: {max_position_error_nn}; Max Orientation Error NN: {max_orientation_error_nn}")




