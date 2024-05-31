import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api import activations as act
from keras.api.layers import Dense
from direct_kin import get_trans_and_rot
import matplotlib.pyplot as plt
from joblib import dump

# Define the number of neurons in each layer
input_size = 6
hidden_layer1_size = 6
hidden_layer2_size = 6
hidden_layer3_size = 6
output_size = 6
number_of_hidden_layers = 0

# Generate some dummy data for demonstration
# input_data = np.random.rand(1000, input_size)
# target_data = np.random.rand(1000, output_size)

data_size = 1000

# Generate the target data. Is is a 1000x6 matrix, where each element is between -2 pi and 2 pi with a seed
np.random.seed(42)
target_data = np.random.rand(data_size, output_size) * 4 * np.pi - 2 * np.pi

input_data = np.ones((data_size, input_size))
for i in range(data_size):
    input_data[i] = get_trans_and_rot(target_data[i])

# normalize the input data using min-max scaler
scaler = MinMaxScaler()
input_data = scaler.fit_transform(input_data)

# normalize the output data using min-max scaler
target_data = scaler.fit_transform(target_data)

# Split the data into training and test sets
input_train, input_test, target_train, target_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

# Create a sequential model
model = Sequential()

# Add the first hidden layer
# model.add(Dense(hidden_layer1_size, input_dim=input_size, activation='relu'))

# Add the second hidden layer
# model.add(Dense(hidden_layer2_size, input_dim=input_size, activation='relu'))

for i in range(number_of_hidden_layers):
    model.add(Dense(hidden_layer2_size, input_dim=input_size, activation=act.relu))

# Add the output layer
model.add(Dense(output_size, activation=act.linear))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model and store the training history with learning rate of 0.01

history = model.fit(input_train, target_train, epochs=100, batch_size=int(data_size*0.01), validation_data=(input_test, target_test))

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

# save the scaler
dump(scaler, 'my_model_scaler.pkl')
