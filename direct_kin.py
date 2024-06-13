import numpy as np
from scipy.spatial.transform import Rotation as R


def get_trans_and_rot(theta_list):
    # if type(theta_list) is list or type(theta_list) is np.ndarray:
    #     iteration_number = len(theta_list)
    # else:
    #     iteration_number = theta_list.shape.dims[0].value
    #
    # # rotation around x-axis
    # alpha_list = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    #
    # # translation along z-axis
    # d_list = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
    #
    # # translation along x-axis
    # a_list = [0, -0.425, -0.39225, 0, 0, 0]
    #
    # trans_list = []
    #
    # for i in range(iteration_number):
    #     current_trans = np.matrix([[np.cos(theta_list[i]), -np.sin(theta_list[i]) * np.cos(alpha_list[i]),
    #                                 np.sin(theta_list[i]) * np.sin(alpha_list[i]), a_list[i] * np.cos(theta_list[i])],
    #                                [np.sin(theta_list[i]), np.cos(theta_list[i]) * np.cos(alpha_list[i]),
    #                                 -np.cos(theta_list[i]) * np.sin(alpha_list[i]), a_list[i] * np.sin(theta_list[i])],
    #                                [0, np.sin(alpha_list[i]), np.cos(theta_list[i]), d_list[i]],
    #                                [0, 0, 0, 1]])
    #
    #     trans_list.append(current_trans)
    #
    # complete_trans = np.eye(4)
    # for transformation in trans_list:
    #     complete_trans = complete_trans.dot(transformation)

    complete_trans = get_homogeneous_transformation(theta_list)

    # Extract translation vector
    translation = complete_trans[:3, 3]

    # Extract rotation matrix
    rotation_matrix = complete_trans[:3, :3]

    # # Convert rotation matrix to Euler angles
    # rotation = R.from_matrix(rotation_matrix)
    #
    # euler_angles = rotation.as_euler('ZYX', degrees=False)  # Change the 'xyz' order if necessary

    z_rot = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    y_rot = np.arcsin(-rotation_matrix[2, 0])
    x_rot = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    euler_angles = np.array([x_rot, y_rot, z_rot])

    # print("Complete transformation matrix: ", complete_trans)
    # print("Translation vector: ", translation)
    # print("Rotation matrix: ", rotation_matrix)
    # print("Euler angles: ", euler_angles)

    # translation_as_list = translation.flatten().tolist()
    # # remove outer brackets
    # translation_as_np = np.array([item for sublist in translation_as_list for item in sublist])

    rot_and_trans = np.concatenate((translation, euler_angles)).reshape(1, 6)
    return rot_and_trans

def get_single_transformation(theta, n):
    # rotation around x-axis
    alpha_list = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

    # translation along z-axis
    d_list = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]

    # translation along x-axis
    a_list = [0, -0.425, -0.39225, 0, 0, 0]

    translation_a = np.identity(4)
    translation_a[0, 3] = a_list[n]

    translation_d = np.identity(4)
    translation_d[2, 3] = d_list[n]

    rotation_z = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    rotation_x = np.array([[1, 0, 0, 0],
                            [0, np.cos(alpha_list[n]), -np.sin(alpha_list[n]), 0],
                            [0, np.sin(alpha_list[n]), np.cos(alpha_list[n]), 0],
                            [0, 0, 0, 1]])

    transformation = np.dot(np.dot(np.dot(translation_d, rotation_z), translation_a), rotation_x)

    return transformation

def get_homogeneous_transformation(theta_list):
    theta_list_reshaped = theta_list

    if type(theta_list_reshaped) is not list:
        if theta_list_reshaped.shape != (6,):
            theta_list_reshaped = theta_list_reshaped.reshape(6)




    A_1 = get_single_transformation(theta_list_reshaped[0], 0)
    A_2 = get_single_transformation(theta_list_reshaped[1], 1)
    A_3 = get_single_transformation(theta_list_reshaped[2], 2)
    A_4 = get_single_transformation(theta_list_reshaped[3], 3)
    A_5 = get_single_transformation(theta_list_reshaped[4], 4)
    A_6 = get_single_transformation(theta_list_reshaped[5], 5)

    T_06 = np.dot(np.dot(np.dot(np.dot(np.dot(A_1, A_2), A_3), A_4), A_5), A_6)

    return T_06


def get_homogeneous_transformation_old(theta_list):
    if type(theta_list) is list or type(theta_list) is np.ndarray:
        iteration_number = len(theta_list)
    else:
        iteration_number = theta_list.shape.dims[0].value

    # rotation around x-axis
    alpha_list = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

    # translation along z-axis
    d_list = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]

    # translation along x-axis
    a_list = [0, -0.425, -0.39225, 0, 0, 0]

    trans_list = []

    for i in range(iteration_number):
        current_trans = np.matrix([[np.cos(theta_list[i]), -np.sin(theta_list[i]) * np.cos(alpha_list[i]),
                                    np.sin(theta_list[i]) * np.sin(alpha_list[i]), a_list[i] * np.cos(theta_list[i])],
                                   [np.sin(theta_list[i]), np.cos(theta_list[i]) * np.cos(alpha_list[i]),
                                    -np.cos(theta_list[i]) * np.sin(alpha_list[i]), a_list[i] * np.sin(theta_list[i])],
                                   [0, np.sin(alpha_list[i]), np.cos(theta_list[i]), d_list[i]],
                                   [0, 0, 0, 1]])

        trans_list.append(current_trans)

    complete_trans = np.eye(4)
    for transformation in trans_list:
        complete_trans = complete_trans.dot(transformation)

    return complete_trans


if __name__ == '__main__':
    # rotation around z-axis
    theta1 = -0.06894051
    theta2 = -1.58825
    theta3 = 1.6226326
    theta4 = 4.69494
    theta5 = -1.6105898
    theta6 = -0.09285152

    theta_list = [theta1, theta2, theta3, theta4, theta5, theta6]
    print(get_trans_and_rot(theta_list))
