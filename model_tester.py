from direct_kin import get_trans_and_rot
import matplotlib.pyplot as plt
from keras.api import models
import numpy as np
import joblib
from custom_loss import custom_loss
from custom_loss import custom_loss_signle_mapping
from plot_data import plot_3d

# load the pre-trained model
model = models.load_model('my_model.keras', custom_objects={'custom_loss': custom_loss})
# model = models.load_model('my_model.keras', custom_objects={'custom_loss_signle_mapping': custom_loss_signle_mapping})

# initialize the MinMaxScaler
#scaler = joblib.load('my_model_scaler.pkl')

# test thetas
#theta_list = [1, 1, 1, 1, 1, 1]
#theta_list = [0.37467555, 0.95108056, 0.73347055, 0.59873642, 0.15611025, 0.1563325]
theta_list = [-0.78828768,  2.83192151,  1.45766093,  0.61988954, -2.16129862, -2.16145018]
trans_and_rot = get_trans_and_rot(theta_list).reshape(1, 6)

# scale the new unseen data
#trans_and_rot_scaled = scaler.transform(trans_and_rot)

# predicted_thetas_scaled = model.predict(trans_and_rot_scaled)
predicted_thetas = model.predict(trans_and_rot)

# inverse transform the predicted thetas to bring them back to the original scale
#predicted_thetas = scaler.inverse_transform(predicted_thetas_scaled)

print("Predicted thetas: ", predicted_thetas)

trans_and_rot_predicted = get_trans_and_rot(predicted_thetas[0])
# trans_and_rot_predicted = get_trans_and_rot(predicted_thetas[0, : -1])
trans_and_rot_original = get_trans_and_rot(theta_list)
loss = np.mean(np.sqrt(np.square(trans_and_rot_original - trans_and_rot_predicted)))
print(f"Predicted rotation and position {str(trans_and_rot_predicted)}")
print(f"Original rotation and position {str(trans_and_rot_original)}")
print(f"Loss: {loss}")

# Plot the original and predicted positions
data_to_plot = np.ones((2, 6))
data_to_plot[0, :] = trans_and_rot_original
data_to_plot[1, :] = trans_and_rot_predicted
plot_3d(data_to_plot)
