import numpy as np
from direct_kin import get_homogeneous_transformation, get_single_transformation, get_trans_and_rot
from symbolic_regression import InverseRegression

def calc_jacobian(thetas):
    Jacobian = np.zeros((6, 6))

    T_0_6 = get_homogeneous_transformation(thetas)  # transformation matrix of the system (forward kinematics)
    point_end = T_0_6[0:3, 3]  # calculate the TCP origin coordinates

    T_0_i = np.array([[1, 0, 0, 0],  # create T_0_0; needed for for-loop
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    for i in range(6):
        if i == 0:  # kinematic chain
            T_0_i = T_0_i  # adds velocity of previous joint to current joint
        else:  # using the DH parameters
            if thetas.shape != (6,):
                theta_list_reshaped = thetas.reshape(6)
            else:
                theta_list_reshaped = thetas
            T = get_single_transformation(theta_list_reshaped[i], i)
            T_0_i = np.dot(T_0_i, T)

        z_i = T_0_i[0:3,
              2]  # gets the vectors p_i and z_i for the Jacobian from the last two coloums of the transformation matrices
        p_i = T_0_i[0:3, 3]
        r = point_end - p_i
        Jacobian[0:3, i] = np.cross(z_i, r)  # linear portion
        Jacobian[3:6,
        i] = z_i  # angular portion             ## each time the loop is passed, another column of the Jacobi matrix is filled

    return Jacobian


def regression(target_pose, initial_theta, max_iterations=1000, learning_rate=1, max_position_error=1e-4, max_orientation_error=0.1):
    jacobian_calculator = InverseRegression()

    if initial_theta.shape != (6, 1):
        initial_theta = initial_theta.reshape(6, 1)

    if target_pose.shape != (6, 1):
        target_pose = target_pose.reshape(6, 1)

    initial_pose = get_trans_and_rot(initial_theta).reshape(6, 1)

    error = np.sqrt(np.sum((target_pose - initial_pose) ** 2))

    iteration = 0
    theta = initial_theta

    position_error = np.sqrt(np.sum((target_pose[:3] - initial_pose[:3]) ** 2))
    orientation_error = np.sqrt(np.sum((target_pose[3:] - initial_pose[3:]) ** 2))

    while (position_error > max_position_error or orientation_error > max_orientation_error) and iteration < max_iterations:
        jacobian = jacobian_calculator.get_jacobian_as_np(theta)  # calculate the Jacobian matrix for the current tar
        jacobian_inv = np.linalg.pinv(jacobian)

        pose = get_trans_and_rot(theta).reshape(6, 1)
        # error = np.sqrt(np.sum((target_pose - pose) ** 2))

        position_error = np.sqrt(np.sum((target_pose[:3] - pose[:3]) ** 2))
        orientation_error = np.sqrt(np.sum((target_pose[3:] - pose[3:]) ** 2))

        if theta.shape != (6, 1):
            theta = theta.reshape(6, 1)

        rot_z = pose[3, 0]
        rot_y = pose[4, 0]
        rot_x = pose[5, 0]

        correction_matrix = np.array([[0, -np.sin(rot_z), np.cos(rot_z) * np.cos(rot_y)],
                                      [0, np.cos(rot_z), np.sin(rot_z) * np.cos(rot_y)],
                                      [1, 0, -np.sin(rot_y)]])

        zeros_3x3 = np.zeros((3, 3))
        eye_3x3 = np.eye(3)
        analytical_to_geometrical_matrix = np.block([[eye_3x3, zeros_3x3], [zeros_3x3, correction_matrix]])
        geometrical_jacobian = np.dot(analytical_to_geometrical_matrix, jacobian)
        geometrical_jacobian_inv = np.linalg.pinv(geometrical_jacobian)

        theta = theta.reshape(6, 1) + learning_rate * np.dot(jacobian_inv, target_pose - pose)
        # theta = theta.reshape(6, 1) - learning_rate * np.dot(geometrical_jacobian_inv, target_pose - pose)

        print("Iteration:", iteration)
        print("Error:", error)
        print(f"Position Error: {position_error}; Orientation Error: {orientation_error}")
        iteration += 1


    return theta, iteration


if __name__ == '__main__':
    theta1 = -0.06894051
    theta2 = -1.58825
    theta3 = 1.6226326
    theta4 = 4.69494
    theta5 = -1.6105898
    theta6 = -0.09285152

    intial_theta = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    changed_theta = np.array([theta1 + 0.2, theta2 + 0.3, theta3 - 0.2, theta4, theta5, theta6])
    target_pose = get_trans_and_rot(intial_theta)

    optimized_theta = regression(target_pose, changed_theta)

    print("Initial Theta:", intial_theta)
    print("Changed Theta:", changed_theta)
    print("Optimized Theta:", optimized_theta)

    optimized_pose = get_trans_and_rot(optimized_theta)

    print("Initial Pose:", target_pose)
    print("Optimized Pose:", optimized_pose)


