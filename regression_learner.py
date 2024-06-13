import numpy as np
import torch

from direct_kin import get_trans_and_rot

def get_forward_pose(thetas):
    def get_single_transformation_tensor(theta, n):
        # rotation around x-axis
        alpha_list = torch.tensor([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0], dtype=torch.float32, requires_grad=True)

        # translation along z-axis
        d_list = torch.tensor([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], dtype=torch.float32, requires_grad=True)

        # translation along x-axis
        a_list = torch.tensor([0, -0.425, -0.39225, 0, 0, 0], dtype=torch.float32, requires_grad=True)

        # translation_a = torch.eye(4)
        # translation_a[0, 3] = a_list[n]

        translation_a = torch.tensor([[1, 0, 0, a_list[n]],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], requires_grad=True)

        # translation_d = torch.eye(4)
        # translation_d[2, 3] = d_list[n]

        translation_d = torch.tensor([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, d_list[n]],
                                        [0, 0, 0, 1]], requires_grad=True)

        rotation_z = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0, 0],
                                   [torch.sin(theta), torch.cos(theta), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], requires_grad=True)

        rotation_x = torch.tensor([[1, 0, 0, 0],
                                   [0, torch.cos(alpha_list[n]), -torch.sin(alpha_list[n]), 0],
                                   [0, torch.sin(alpha_list[n]), torch.cos(alpha_list[n]), 0],
                                   [0, 0, 0, 1]], requires_grad=True)

        transformation = torch.mm(torch.mm(torch.mm(translation_d, rotation_z), translation_a), rotation_x)

        return transformation

    A_1 = get_single_transformation_tensor(thetas[0], 0)
    A_2 = get_single_transformation_tensor(thetas[1], 1)
    A_3 = get_single_transformation_tensor(thetas[2], 2)
    A_4 = get_single_transformation_tensor(thetas[3], 3)
    A_5 = get_single_transformation_tensor(thetas[4], 4)
    A_6 = get_single_transformation_tensor(thetas[5], 5)

    complete_trans = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(A_1, A_2), A_3), A_4), A_5), A_6)

    # Extract translation vector
    translation = complete_trans[:3, 3]
    rotation_matrix = complete_trans[:3, :3]

    # Convert rotation matrix to Euler angles
    z_rot = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # y_rot = torch.atan2(-rotation_matrix[2, 0], torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    y_rot = torch.asin(-rotation_matrix[2, 0])
    x_rot = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    print("Calculated Rotation:")
    print(x_rot, y_rot, z_rot)

    pose = torch.cat((translation, torch.tensor([x_rot, y_rot, z_rot])), 0)

    return pose

def calucalte_jacobian(thetas):
    # Calculate the forward pose
    thetas_w_grad = thetas.requires_grad_(True)

    pose = get_forward_pose(thetas_w_grad).reshape(6, 1)

    # Calculate the Jacobian
    jacobian = torch.autograd.functional.jacobian(get_forward_pose, thetas_w_grad)
    print("Jacobian:", jacobian)

    return jacobian

def reduce_error(target_pose, initial_theta, max_error=1e-6, max_iterations=1000):
    # Initial error
    error = np.inf

    target_pose = target_pose.reshape(6, 1)

    target_tensor = torch.tensor(target_pose, requires_grad=True)

    # Initial theta
    theta = torch.tensor(initial_theta, requires_grad=True).reshape(6, 1)

    pose = get_forward_pose(theta).reshape(6, 1)

    error = torch.norm(target_tensor - pose)

    print("Initial Error:", error)

    # Number of iterations
    iterations = 0

    while error > max_error and iterations < max_iterations:
        # Calculate the error
        pose = get_forward_pose(theta).reshape(6, 1)

        old_error = torch.norm(target_tensor - pose)

        # Calculate the Jacobian
        thetas = torch.tensor([theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]]).reshape(6, 1)
        jacobian_tensor = calucalte_jacobian(thetas)

        # Calculate the pseudo-inverse of the Jacobian
        jacobian_pinv = torch.linalg.pinv(jacobian_tensor)

        # Calculate the change in theta
        delta_theta = torch.mm(jacobian_pinv, old_error)

        # Update theta
        theta += delta_theta
        error = torch.norm(target_tensor - pose)
        # Increase the number of iterations
        iterations += 1

        print(f"Iteration: {iterations}, Error: {old_error}")

    return theta



if __name__ == '__main__':
    # Beispielzielpose (x, y, z, roll, pitch, yaw)
    # target_pose = np.array([0.38241358, -0.36891149, 0.23293635, 2.25538142, 0.26250541, 2.81564745])

    theta1 = -0.06894051
    theta2 = -1.58825
    theta3 = 1.6226326
    theta4 = 4.69494
    theta5 = -1.6105898
    theta6 = -0.09285152


    t1 = torch.tensor([theta1], requires_grad=True)
    t2 = torch.tensor([theta2], requires_grad=True)
    t3 = torch.tensor([theta3], requires_grad=True)
    t4 = torch.tensor([theta4], requires_grad=True)
    t5 = torch.tensor([theta5], requires_grad=True)
    t6 = torch.tensor([theta6], requires_grad=True)

    # t1 = torch.tensor([0.38241358], requires_grad=True)
    # t2 = torch.tensor([-0.36891149], requires_grad=True)
    # t3 = torch.tensor([0.23293635], requires_grad=True)
    # t4 = torch.tensor([2.25538142], requires_grad=True)
    # t5 = torch.tensor([0.26250541], requires_grad=True)
    # t6 = torch.tensor([2.81564745], requires_grad=True)

    target_pose = get_trans_and_rot([theta1 + 0.1, theta2, theta3, theta4, theta5, theta6])
    print("Target Pose:", target_pose)

    new_thetas = reduce_error(target_pose, [t1, t2, t3, t4, t5, t6])

    print("New thetas:", new_thetas)

    new_pose_tensor = get_forward_pose(new_thetas)

    exit()

    # Initiale Schätzung der Gelenkwinkel (in Rad)
    initial_theta = np.array([1.2095332, -0.26787275, -2.7469535, 2.330062, 0.9455628, 0.8073318])

    # Finde die Gelenkwinkel, die die Zielpose erreichen
    optimized_theta = newton_raphson_ik(target_pose, initial_theta)

    print("Optimierte Gelenkwinkel:", optimized_theta)

    # Berechne die Endeffektorpose mit den optimierten Gelenkwinkeln
    optimized_pose = forward_kinematics(optimized_theta.tolist())

    print("Optimierte Endeffektorpose:", optimized_pose)
