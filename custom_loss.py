
import numpy as np
import tensorflow as tf

def get_rot_and_trans_tensor(theta_list):
    def get_signle_transformation_tensor(theta, n):
        alpha_list = tf.constant([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0], dtype=tf.float32)
        d_list = tf.constant([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], dtype=tf.float32)
        a_list = tf.constant([0, -0.425, -0.39225, 0, 0, 0], dtype=tf.float32)

        translation_a = tf.eye(4)
        translation_a = tf.tensor_scatter_nd_update(translation_a, [[0, 3]], [a_list[n]])

        translation_d = tf.eye(4)
        translation_d = tf.tensor_scatter_nd_update(translation_d, [[2, 3]], [d_list[n]])

        rotation_z = tf.stack([
            tf.stack([tf.cos(theta), -tf.sin(theta), tf.zeros_like(theta), tf.zeros_like(theta)], axis=-1),
            tf.stack([tf.sin(theta), tf.cos(theta), tf.zeros_like(theta), tf.zeros_like(theta)], axis=-1),
            tf.stack([tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta), tf.zeros_like(theta)], axis=-1),
            tf.stack([tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)], axis=-1)
        ], axis=-2)

        rotation_x = tf.stack([
            tf.stack([tf.ones_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta)], axis=-1),
            tf.stack([tf.zeros_like(theta), tf.fill(tf.shape(theta), tf.cos(alpha_list[n])), tf.fill(tf.shape(theta), -tf.sin(alpha_list[n])), tf.zeros_like(theta)], axis=-1),
            tf.stack([tf.zeros_like(theta), tf.fill(tf.shape(theta), tf.sin(alpha_list[n])), tf.fill(tf.shape(theta), tf.cos(alpha_list[n])), tf.zeros_like(theta)], axis=-1),
            tf.stack([tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)], axis=-1)
        ], axis=-2)

        transformation = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.matmul(translation_d, rotation_z), translation_a), rotation_x)

        return transformation

    batch_size = tf.shape(theta_list)[0]

    A_1 = get_signle_transformation_tensor(theta_list[:, 0], 0)
    A_2 = get_signle_transformation_tensor(theta_list[:, 1], 1)
    A_3 = get_signle_transformation_tensor(theta_list[:, 2], 2)
    A_4 = get_signle_transformation_tensor(theta_list[:, 3], 3)
    A_5 = get_signle_transformation_tensor(theta_list[:, 4], 4)
    A_6 = get_signle_transformation_tensor(theta_list[:, 5], 5)

    complete_trans = tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf.matmul(A_1, A_2), A_3), A_4), A_5), A_6)

    translation = complete_trans[:, :3, 3]
    rotation = complete_trans[:, :3, :3]

    z_rot = tf.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    y_rot = tf.asin(-rotation[:, 2, 0])
    x_rot = tf.atan2(rotation[:, 2, 1], rotation[:, 2, 2])

    trans_and_rot = tf.concat([translation, tf.stack([x_rot, y_rot, z_rot], axis=-1)], axis=-1)

    return trans_and_rot



# def get_rot_and_trans_tensor_old(theta_list):
#     alpha_list = tf.constant([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0], dtype=tf.float32)
#     d_list = tf.constant([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], dtype=tf.float32)
#     a_list = tf.constant([0, -0.425, -0.39225, 0, 0, 0], dtype=tf.float32)
#
#     batch_size = tf.shape(theta_list)[0]
#
#     complete_trans = tf.eye(4, batch_shape=[batch_size])
#     for i in range(6):
#         theta = theta_list[:, i]
#         alpha = alpha_list[i]
#         d = d_list[i]
#         a = a_list[i]
#
#         zeros = tf.zeros_like(theta)
#         ones = tf.ones_like(theta)
#
#         current_trans = tf.stack([
#             tf.stack([tf.cos(theta), -tf.sin(theta) * tf.cos(alpha), tf.sin(theta) * tf.sin(alpha), a * tf.cos(theta)], axis=-1),
#             tf.stack([tf.sin(theta), tf.cos(theta) * tf.cos(alpha), -tf.cos(theta) * tf.sin(alpha), a * tf.sin(theta)], axis=-1),
#             tf.stack([zeros, tf.fill(tf.shape(theta), tf.sin(alpha)), tf.fill(tf.shape(theta), tf.cos(alpha)), tf.fill(tf.shape(theta), d)], axis=-1),
#             tf.stack([zeros, zeros, zeros, ones], axis=-1)
#         ], axis=-2)
#
#         complete_trans = tf.linalg.matmul(complete_trans, current_trans)
#
#     translation = complete_trans[:, :3, 3]
#     rotation_matrix = complete_trans[:, :3, :3]
#
#     def rotation_matrix_to_euler_angles(matrix):
#         sy = tf.sqrt(matrix[:, 0, 0] ** 2 + matrix[:, 1, 0] ** 2)
#         singular = sy < 1e-6
#
#         x = tf.where(singular, tf.atan2(-matrix[:, 1, 2], matrix[:, 1, 1]), tf.atan2(matrix[:, 2, 1], matrix[:, 2, 2]))
#         y = tf.where(singular, tf.atan2(-matrix[:, 2, 0], sy), tf.atan2(-matrix[:, 2, 0], sy))
#         z = tf.where(singular, tf.zeros_like(matrix[:, 2, 0]), tf.atan2(matrix[:, 1, 0], matrix[:, 0, 0]))
#
#         return tf.stack([x, y, z], axis=1)
#
#     euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
#
#     rot_and_trans = tf.concat([translation, euler_angles], axis=-1)
#     return rot_and_trans

def custom_loss(y_true, y_pred):
    y_true_rot_trans = get_rot_and_trans_tensor(y_true)
    y_pred_rot_trans = get_rot_and_trans_tensor(y_pred)

    loss = tf.reduce_mean(tf.square(y_true_rot_trans - y_pred_rot_trans))
    return loss

def custom_loss_signle_mapping(y_true, y_pred):
    joints_true = y_true[:, :-1]
    joints_pred = y_pred[:, :-1]

    last_column_true = y_true[:, -1]
    last_column_pred = y_pred[:, -1]

    y_true_rot_trans = get_rot_and_trans_tensor(joints_true)
    y_pred_rot_trans = get_rot_and_trans_tensor(joints_pred)

    # add the last column to the rotation and translation
    y_true = tf.concat([y_true_rot_trans, last_column_true[:, tf.newaxis]], axis=-1)
    y_pred = tf.concat([y_pred_rot_trans, last_column_pred[:, tf.newaxis]], axis=-1)

    loss = tf.reduce_mean(tf.square(y_true - y_pred))

    return loss