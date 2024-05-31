import numpy as np
from keras.api.models import Model
from keras.api.layers import Input, Dense
from keras.api.optimizers import Adam

# Define the dimensions
input_dim = 6  # x, y, z, and 3 Euler angles
encoding_dim = 3  # Size of the latent space (encoded representation)

# Define the encoder
input_pose = Input(shape=(input_dim, ))
encoded = Dense(128, activation='relu')(input_pose)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Define the decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Define the autoencoder
autoencoder = Model(input_pose, decoded)

# Define the encoder model to extract features
encoder = Model(input_pose, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Example data (replace with your actual data)
X_train = np.load("input_data.npy")

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2)

# Extract the encoded features
encoded_poses = encoder.predict(X_train)
print(encoded_poses)