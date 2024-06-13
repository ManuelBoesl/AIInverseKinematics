import sympy as sym
import numpy as np

class InverseRegression:
    def __init__(self):
        self.symbolic_joints_list = [sym.Symbol('q1'), sym.Symbol('q2'), sym.Symbol('q3'), sym.Symbol('q4'), sym.Symbol('q5'), sym.Symbol('q6')]
        # rotation around x-axis
        self.derivative_matrix = sym.Matrix(self.symbolic_joints_list)

        self.alpha_list = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

        # translation along z-axis
        self.d_list = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]

        # translation along x-axis
        self.a_list = [0, -0.425, -0.39225, 0, 0, 0]

    def get_single_transformation(self, n):
        translation_a = sym.eye(4)
        translation_a[0, 3] = self.a_list[n]

        translation_d = sym.eye(4)
        translation_d[2, 3] = self.d_list[n]

        rotation_z = sym.Matrix([[sym.cos(self.symbolic_joints_list[n]), -sym.sin(self.symbolic_joints_list[n]), 0, 0],
                                [sym.sin(self.symbolic_joints_list[n]), sym.cos(self.symbolic_joints_list[n]), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        rotation_x = sym.Matrix([[1, 0, 0, 0],
                                [0, sym.cos(self.alpha_list[n]), -sym.sin(self.alpha_list[n]), 0],
                                [0, sym.sin(self.alpha_list[n]), sym.cos(self.alpha_list[n]), 0],
                                [0, 0, 0, 1]])

        transformation = translation_d * rotation_z * translation_a * rotation_x

        return transformation

    def get_homogeneous_transformation(self):
        A_1 = self.get_single_transformation(0)
        A_2 = self.get_single_transformation(1)
        A_3 = self.get_single_transformation(2)
        A_4 = self.get_single_transformation(3)
        A_5 = self.get_single_transformation(4)
        A_6 = self.get_single_transformation(5)

        T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6

        return T_06

    def get_trans_and_rot(self):
        T_06 = self.get_homogeneous_transformation()

        translation = T_06[:3, 3]
        rotation = T_06[:3, :3]

        z_rot = sym.atan2(rotation[1, 0], rotation[0, 0])
        y_rot = sym.asin(-rotation[2, 0])
        x_rot = sym.atan2(rotation[2, 1], rotation[2, 2])

        euler_angles = sym.Matrix([x_rot, y_rot, z_rot])

        trans_and_rot = sym.Matrix([translation, euler_angles])

        return trans_and_rot

    def get_jacobian(self):
        trans_and_rot = self.get_trans_and_rot()
        jacobian = trans_and_rot.jacobian(self.derivative_matrix)

        return jacobian

    def get_jacobian_as_np(self, theta_list):
        jacobian = self.get_jacobian()

        replaced_theta_list = None

        if type(theta_list) is not list:
            replaced_theta_list = []
        else:
            replaced_theta_list = theta_list

        for i in range(6):
            if type(theta_list) is not list:
                if theta_list.shape == (6,):
                    replaced_theta_list.append(theta_list[i])
                else:
                    replaced_theta_list.append(theta_list[i, 0])

        replaced_jacobian = jacobian.subs({self.symbolic_joints_list[0]: replaced_theta_list[0], self.symbolic_joints_list[1]: replaced_theta_list[1], self.symbolic_joints_list[2]: replaced_theta_list[2],
               self.symbolic_joints_list[3]: replaced_theta_list[3], self.symbolic_joints_list[4]: replaced_theta_list[4], self.symbolic_joints_list[5]: replaced_theta_list[5]})
        jacobian_as_np = np.array(replaced_jacobian).astype(np.float64)

        return jacobian_as_np


if __name__ == '__main__':
    theta1 = -0.06894051
    theta2 = -1.58825
    theta3 = 1.6226326
    theta4 = 4.69494
    theta5 = -1.6105898
    theta6 = -0.09285152

    ir = InverseRegression()
    pose = ir.get_trans_and_rot()
    print(pose)

    replaced_pose = pose.subs({ir.symbolic_joints_list[0]: theta1, ir.symbolic_joints_list[1]: theta2, ir.symbolic_joints_list[2]: theta3,
               ir.symbolic_joints_list[3]: theta4, ir.symbolic_joints_list[4]: theta5, ir.symbolic_joints_list[5]: theta6})

    print(replaced_pose)

    matrix_as_np = np.array(replaced_pose).astype(np.float64)

    print(matrix_as_np)
