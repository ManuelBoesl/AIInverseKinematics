import numpy as np
import tensorflow as tf
from keras.api.layers import Dense
from sklearn.model_selection import train_test_split
from direct_kin import get_trans_and_rot
from scipy.spatial.transform import Rotation as R
import keras

# Define your neural network
class MyNeuralNetwork(keras.Model):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.input_layer = Dense(6, activation='tanh')
        self.hidden_layer = Dense(6, activation='tanh')
        self.output_layer = Dense(6)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        return self.output_layer(x)


# Generate the target data
num_epochs = 100
output_size = 6
data_size = 1000
np.random.seed(42)
target_data = np.random.rand(data_size, output_size) * 2 * np.pi - np.pi

# Generate the input data using the target data
input_data = np.ones((data_size, 6))
for i in range(data_size):
    input_data[i] = get_trans_and_rot(target_data[i])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

# Create an instance of the neural network
model = MyNeuralNetwork()


def get_rot_and_trans_tensor(theta_list):
    # Define constants
    alpha_list = tf.constant([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0], dtype=tf.float32)
    d_list = tf.constant([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], dtype=tf.float32)
    a_list = tf.constant([0, -0.425, -0.39225, 0, 0, 0], dtype=tf.float32)

    # Reshape theta_list if necessary
    if theta_list.ndim == 1:
        theta_list = tf.expand_dims(theta_list, axis=0)

    # Calculate transformation
    trans_list = []
    for theta in theta_list:
        rotation = tf.reshape(
            tf.stack([tf.cos(theta), -tf.sin(theta) * tf.cos(alpha_list), tf.sin(theta) * tf.sin(alpha_list)], axis=1),
            (-1, 3, 3))
        translation = tf.stack([tf.stack([a * tf.cos(theta), a * tf.sin(theta), tf.broadcast_to(d, tf.shape(theta))]) for a, d in zip(a_list, d_list)])
        translation = tf.concat([translation, tf.constant([[0., 0., 0.]], dtype=tf.float32)], axis=0)
        trans = tf.concat([rotation, tf.transpose(tf.expand_dims(translation, axis=-1), perm=[0, 2, 1])], axis=1)
        trans = tf.concat([trans, tf.constant([[[0., 0., 0., 1.]]], dtype=tf.float32)], axis=0)
        trans_list.append(trans)

    # Compute complete transformation
    complete_trans = tf.eye(4, batch_shape=[tf.shape(theta_list)[0]], dtype=tf.float32)
    for trans in trans_list:
        complete_trans = tf.matmul(complete_trans, trans)

    # Extract translation vector
    translation = tf.squeeze(complete_trans[:, :3, 3])

    # Extract rotation matrix
    rotation_matrix = complete_trans[:, :3, :3]

    # Convert rotation matrix to Euler angles
    rotation = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)  # Change the 'xyz' order if necessary

    return tf.concat([translation, rotation], axis=1)


def compute_loss(predicted_joint_angles, target_joint_angles):
    predicted_poses = get_rot_and_trans_tensor(predicted_joint_angles)
    target_poses = get_rot_and_trans_tensor(target_joint_angles)
    return tf.reduce_mean(tf.square(predicted_poses - target_poses))


# Define your optimizer
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

# Train the network
for epoch in range(num_epochs):
    # Convert your data to TensorFlow tensors
    inputs = tf.constant(X_train, dtype=tf.float32)
    targets = tf.constant(y_train, dtype=tf.float32)

    # Forward pass
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = compute_loss(outputs, targets)

    # Backward pass and optimization
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Validate the model on the validation set
    if (epoch + 1) % 10 == 0:
        val_inputs = tf.constant(X_val, dtype=tf.float32)
        val_targets = tf.constant(y_val, dtype=tf.float32)
        val_outputs = model(val_inputs)
        val_loss = compute_loss(val_outputs, val_targets)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss:.8f}, Validation Loss: {val_loss:.8f}')
