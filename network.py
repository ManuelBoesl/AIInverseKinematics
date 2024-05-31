import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from direct_kin import get_trans_and_rot
from scipy.spatial.transform import Rotation as R


# Define your neural network
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(6, 6)  # 6 neurons in, 6 neurons out
        self.hidden_layer = nn.Linear(6, 6)  # 6 neurons in, 6 neurons out
        self.output_layer = nn.Linear(6, 6)  # 6 neurons in, 6 neurons out

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))  # Apply tanh activation to input layer
        x = torch.tanh(self.hidden_layer(x))  # Apply tanh activation to hidden layer
        x = self.output_layer(x)  # Output layer with linear activation
        return x


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
    # if isinstance(theta_list, list) or isinstance(theta_list, np.ndarray):
    #     theta_list = torch.tensor(theta_list, dtype=torch.float32)

    if theta_list.dim() == 1:
        iteration_number = 1
    else:
        iteration_number = theta_list.size(0)

    # rotation around x-axis
    alpha_list = torch.tensor([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0], dtype=torch.float32)

    # translation along z-axis
    d_list = torch.tensor([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], dtype=torch.float32)

    # translation along x-axis
    a_list = torch.tensor([0, -0.425, -0.39225, 0, 0, 0], dtype=torch.float32)

    trans_list = []

    for i in range(iteration_number):
        current_trans = torch.tensor([[torch.cos(theta_list[i]), -torch.sin(theta_list[i]) * torch.cos(alpha_list[i]),
                                       torch.sin(theta_list[i]) * torch.sin(alpha_list[i]),
                                       a_list[i] * torch.cos(theta_list[i])],
                                      [torch.sin(theta_list[i]), torch.cos(theta_list[i]) * torch.cos(alpha_list[i]),
                                       -torch.cos(theta_list[i]) * torch.sin(alpha_list[i]),
                                       a_list[i] * torch.sin(theta_list[i])],
                                      [0, torch.sin(alpha_list[i]), torch.cos(theta_list[i]), d_list[i]],
                                      [0, 0, 0, 1]], dtype=torch.float32)

        trans_list.append(current_trans)

    complete_trans = torch.eye(4, dtype=torch.float32)
    for transformation in trans_list:
        complete_trans = complete_trans.mm(transformation)

    # Extract translation vector
    translation = complete_trans[:3, 3]

    # Extract rotation matrix
    rotation_matrix = complete_trans[:3, :3]

    # Convert rotation matrix to Euler angles
    rotation = torch.from_numpy(
        R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False))  # Change the 'xyz' order if necessary

    translation_as_list = translation.flatten().tolist()
    # remove outer brackets
    # translation_as_np = torch.tensor([item for sublist in translation_as_list for item in sublist], dtype=torch.float32)

    trans_and_rot = torch.cat((translation, rotation))
    return trans_and_rot


def compute_loss(predicted_joint_angles, target_joint_angles):
    # Calculate loss based on the discrepancy between predictions and targets
    target_poses = torch.ones((targets.size(0), targets.size(1)), dtype=torch.float32)
    predicted_poses = torch.ones((predicted_joint_angles.size(0), predicted_joint_angles.size(1)), dtype=torch.float32)

    for i in range(predicted_joint_angles.size(0)):
        predicted_pose = get_rot_and_trans_tensor(predicted_joint_angles[i])
        predicted_poses[i] = predicted_pose

        target_poses[i] = get_rot_and_trans_tensor(target_joint_angles[i])

    return torch.mean(torch.square(predicted_poses - target_poses))


# Define your optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Train the network
for epoch in range(num_epochs):
    # Convert your data to torch tensors
    inputs = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)

    # Forward pass
    outputs = model(inputs)
    outputs.requires_grad_()

    # Convert outputs to numpy array
    # predictions = outputs.detach().numpy()

    # Calculate the loss
    # loss = compute_loss(predictions, targets)
    loss = compute_loss(outputs, targets)
    loss.requires_grad_()
    # Backward pass and optimization
    #optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagation
    # loss_tensor = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
    # loss_tensor.backward()  # Backpropagation
    optimizer.step()  # Update weights

    # Validate the model on the validation set
    if (epoch + 1) % 10 == 0:
        val_inputs = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
        val_targets = torch.tensor(y_val, dtype=torch.float32, requires_grad=True)
        val_outputs = model(val_inputs)
        val_loss = compute_loss(val_outputs, val_targets)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss:.8f}, Validation Loss: {val_loss:.8f}')
