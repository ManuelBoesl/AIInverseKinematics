import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api import activations as act
from keras.api.layers import Dense
from torch.nn.modules import activation

from direct_kin import get_trans_and_rot
import matplotlib.pyplot as plt
from custom_loss import custom_loss_signle_mapping
from plot_data import plot_3d
from keras.api.utils import plot_model
import tensorflow as tf
from keras.api.utils import set_random_seed
from keras.api import Input


# y_true[i].shape.dims[0].value

# Define the number of neurons in each layer
input_size = 6
hidden_layer_size = 6
output_size = 7
number_of_hidden_layers = 2
number_of_epochs = 300

# Generate some dummy data for demonstration
# input_data = np.random.rand(1000, input_size)
# target_data = np.random.rand(1000, output_size)



# Generate the target data. Is is a 1000x6 matrix, where each element is between -pi and pi with a seed
np.random.seed(42)
set_random_seed(42)

# load the training and target data
input_data = np.load('input_data.npy')
target_data = np.load('target_data.npy')

# add a column to the target data in the last column which is all the elements of the same row summed up
# target_data_single_mapping = np.append(target_data, np.product(input_data, axis=1).reshape(-1, 1), axis=1)
target_data_single_mapping = np.append(target_data, np.sum(input_data, axis=1).reshape(-1, 1), axis=1)

# plot the position of the input data in a 3d plot

# for i in range(data_size):
#     # For every input data, calculate the distance to the whole dataset and replace it if it is too close
#     for j in range(data_size):
#         if i != j:
#             while np.linalg.norm(input_data[i] - input_data[j]) < 2:
#                 target_data[i] = np.random.rand(output_size) * 2 * np.pi - np.pi
#                 input_data[i] = get_rot_and_trans(target_data[i])
#                 print("Replaced data at index ", i)

# target_data, input_data = generate_training_joints()

# normalize the input data using min-max scaler
# scaler = MinMaxScaler()
# input_data = scaler.fit_transform(input_data)

# normalize the output data using min-max scaler
# target_data = scaler.fit_transform(target_data)

# Split the data into training and test sets
input_train, input_test, target_train, target_test = train_test_split(input_data, target_data_single_mapping, test_size=0.2, random_state=42)

print("First row of the target data: ", target_train[0])


# Create a sequential model
model = Sequential()

# Add the first hidden layer
# model.add(Dense(hidden_layer1_size, input_dim=input_size, activation='relu'))

# Add the second hidden layer
# model.add(Dense(hidden_layer2_size, input_dim=input_size, activation='relu'))
# model.add(Dense(hidden_layer_size, input_dim=input_size, activation=act.relu, name='input_layer'))

Input(shape=(1, input_size))

for i in range(number_of_hidden_layers):
    hidden_layer_name = f'hidden_layer_{i}'
    # model.add(Dense(hidden_layer_size, input_dim=input_size, activation=act.tanh, name=hidden_layer_name))
    model.add(Dense(hidden_layer_size,  activation=act.tanh, name=hidden_layer_name))

# Add the output layer
model.add(Dense(output_size, activation=act.linear, name='output_layer'))

# Compile the model
model.compile(loss=custom_loss_signle_mapping, optimizer='adam')

# Train the model and store the training history with learning rate of 0.01

history = model.fit(input_train, target_train, epochs=number_of_epochs, batch_size=100, validation_data=(input_test, target_test))

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Save the model
model.save('my_model.keras')

# Plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# save the scaler
#dump(scaler, 'my_model_scaler.pkl')

print("First row of the target data: ", target_data_single_mapping[0])
