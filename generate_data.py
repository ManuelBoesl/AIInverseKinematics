from direct_kin import get_trans_and_rot
import numpy as np
from plot_data import plot_3d
from scipy.spatial import KDTree

def generate_pseudo_data(data_size, output_size, input_size):
    # Generate the target data. Is is a 1000x6 matrix, where each element is 1
    target_data = np.ones((data_size, output_size))
    input_data = np.ones((data_size, input_size))
    for i in range(data_size):
        input_data[i] = get_trans_and_rot(target_data[i])

    np.save('input_data.npy', input_data)
    np.save('target_data.npy', target_data)

def generate_data(data_size, output_size, input_size):
    # Generate the target data. Is is a 1000x6 matrix, where each element is between -pi and pi with a seed
    np.random.seed(42)
    target_data = np.random.rand(data_size, output_size) * 2 * np.pi - np.pi

    input_data = np.ones((data_size, input_size))
    for i in range(data_size):
        input_data[i] = get_trans_and_rot(target_data[i])

    input_data, target_data = clean_data(input_data, target_data)
    # input_data, target_data = clean_data(target_data)

    print("Data generated successfully")

    # save the data as a numpy file
    np.save('input_data.npy', input_data)
    np.save('target_data.npy', target_data)

    print("Data saved successfully")

    plot_3d(input_data)

    print("Data plotted successfully")


def clean_data(input_data, target_data):
    # This method cleanes the data recursively until all the data points are at least 2 units apart
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[0]):
            if i != j:
                replaced = False
                while np.linalg.norm(input_data[i] - input_data[j]) < 0.5:
                    target_data[i] = np.random.rand(target_data.shape[1]) * 2 * np.pi - np.pi
                    input_data[i] = get_trans_and_rot(target_data[i])
                    print("Replaced data at index ", i)
                    replaced = True
                if replaced:
                    input_data, target_data = clean_data(input_data, target_data)
    return input_data, target_data


# def clean_data(input_data, target_data, min_distance=0.5):
#     """
#     This method ensures all the data points are at least min_distance units apart.
#     Only the position (first three elements of input_data) is considered for distance calculation.
#     """
#     # input_data = np.array([get_trans_and_rot(theta) for theta in target_data])
#     positions = input_data[:, :3]  # Extract positions (first 3 columns)
#
#     accepted_points = []
#     accepted_targets = []
#     tree = KDTree(positions)
#
#     for i, pos in enumerate(positions):
#         if len(accepted_points) == 0:
#             accepted_points.append(pos)
#             accepted_targets.append(target_data[i])
#             continue
#
#         # Query the KD-tree for the nearest neighbor within min_distance
#         distances, _ = tree.query(pos, k=1, distance_upper_bound=min_distance)
#
#         if distances > min_distance:
#             accepted_points.append(pos)
#             accepted_targets.append(target_data[i])
#             tree = KDTree(np.array(accepted_points))
#
#     return np.array(accepted_points), np.array(accepted_targets)

if __name__ == '__main__':
    data_size = 600
    input_size = 6
    output_size = 6
    generate_data(data_size, output_size, input_size)
    # generate_pseudo_data(data_size, output_size, input_size)
