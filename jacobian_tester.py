def jacobian_matrix(q: np.ndarray, dh_para: np.ndarray, runden = False) -"""insert spitze klammer here""" np.array:       # code in the video decription

    Jacobian = np.zeros((6, 6))

    T_0_6 = fk_ur(q, dh_para)               # transformation matrix of the system (forward kinematics)
    point_end = T_0_6[0:3, 3]               # calculate the TCP origin coordinates

    T_0_i = np.array([[1, 0, 0, 0],         # create T_0_0; needed for for-loop
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    for i in range(6):

        if i == 0:                          # kinematic chain
            T_0_i = T_0_i                   # adds velocity of previous joint to current joint
        else:                               # using the DH parameters
            T = dh(dh_para[i-1, 0], dh_para[i-1, 1], dh_para[i-1, 2], q[i-1])
            T_0_i = np.dot(T_0_i, T)

        z_i = T_0_i[0:3, 2]                 # gets the vectors p_i and z_i for the Jacobian from the last two coloums of the transformation matrices  
        p_i = T_0_i[0:3, 3]
        r = point_end - p_i
        Jacobian[0:3, i] = np.cross(z_i, r) # linear portion
        Jacobian[3:6, i] = z_i              # angular portion             ## each time the loop is passed, another column of the Jacobi matrix is filled

        if runden:
            Jacobian[0:3, i] = np.round(np.cross(z_i, r), 3)              # round if True
            Jacobian[3:6, i] = np.round(z_i, 3)

    return Jacobian

def fk_ur(q: np.ndarray, dh_para: np.ndarray) - np.ndarray:
    """
    Forward Kinematics for UR type robots
    :param: dh_para: DH-Transformation, table of dh parameters (alpha, a, d, theta)
    :param: q: Gelenkwinkel
    """
    T_0_6 = np.zeros((4, 4))

    dh_params_count = dh_para.shape[1]
    number_dh_trafos = dh_para.shape[0]

    if dh_params_count != 4:
        print("Wrong number of dh parameters!")
        return None

    trafo_matrizes = []

    for i in range(number_dh_trafos):
        trafo_matrizes.append(dh(dh_para[i, 0], dh_para[i, 1], dh_para[i, 2], q[i]))

    if len(trafo_matrizes) != 0:
        for i in range(len(trafo_matrizes) - 1):
            if i == 0:
                T_0_6 = trafo_matrizes[i] @ trafo_matrizes[i+1]
            else:
                T_0_6 = T_0_6 @ trafo_matrizes [i+1]

    return T_0_6


- „Testing“:

print("STARTING: ", datetime.now())