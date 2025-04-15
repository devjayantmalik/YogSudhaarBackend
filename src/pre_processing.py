import matplotlib.pyplot as plt
# import params
import numpy as np

from . import params

plot_list = ["lH_lK_lA", "rH_rK_rA", "rS_rH_rK", "lS_lE_lW", "rS_rE_rW", "lS_lH_lK"]

# 0 - nose
# 1 - left eye (inner)
# 2 - left eye
# 3 - left eye (outer)
# 4 - right eye (inner)
# 5 - right eye
# 6 - right eye (outer)
# 7 - left ear
# 8 - right ear
# 9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index
# 32 - right foot index

sheet_to_df_map_processed2 = {}
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                    (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                    (29, 31), (30, 32), (27, 31), (28, 32)]
rel = POSE_CONNECTIONS
kde_func = []


def plot_stick(df):
    df = df.reset_index()
    for pos in range(len(df)):
        positions_x = []
        ax = 0
        ay = 1
        positions_y = []
        for a, b in rel:
            a = a
            b = b
            mult_factor = -1
            positions_x.extend([df.loc[pos, 'x' + str(a)].item(), df.loc[pos, 'x' + str(b)].item(), None])
            positions_y.extend(
                [mult_factor * df.loc[pos, 'y' + str(a)].item(), mult_factor * df.loc[pos, 'y' + str(b)].item(), None])

        plt.plot(positions_x, positions_y)
        plt.show()


def py_ang(point_a, point_b):
    # if len(np.ravel(point_a)) == 2:
    #     x = point_a[0]
    #     y = point_a[1]
    # else:
    x = [row[0] for row in point_a]
    y = [row[1] for row in point_a]

    ang_a = np.arctan2(y, x)

    # if len(np.ravel(point_b)) == 2:
    #     x = point_b[0]
    #     y = point_b[1]
    # else:
    x = [row[0] for row in point_b]
    y = [row[1] for row in point_b]

    ang_b = np.arctan2(y, x)
    return np.rad2deg((ang_a - ang_b) % (2 * np.pi))


def get_angle(A, B, C, centered_filtered, pos=None):
    coords_ids = params.coords_ids
    A = str(coords_ids[params.keys.index(A)])
    B = str(coords_ids[params.keys.index(B)])
    C = str(coords_ids[params.keys.index(C)])
    p_A = np.array([centered_filtered.loc[:, "x" + A].values, centered_filtered.loc[:, "y" + A].values]).T
    p_B = np.array([centered_filtered.loc[:, "x" + B].values, centered_filtered.loc[:, "y" + B].values]).T
    p_C = np.array([centered_filtered.loc[:, "x" + C].values, centered_filtered.loc[:, "y" + C].values]).T
    p_BA = p_A - p_B
    p_BC = p_C - p_B
    return py_ang(p_BA, p_BC)


def angle_from_x(A, B, centered_filtered, pos=None):
    coords_ids = params.coords_ids
    A = str(coords_ids[params.keys.index(A)])
    B = str(coords_ids[params.keys.index(B)])
    p_A = np.array([centered_filtered.loc[:, "x" + A].values, centered_filtered.loc[:, "y" + A].values]).T
    p_B = np.array([centered_filtered.loc[:, "x" + B].values, centered_filtered.loc[:, "y" + B].values]).T
    point_a = p_A - p_B

    if len(np.ravel(point_a)) == 2:
        x = point_a[0]
        y = point_a[1]
    else:
        x = [row[0] for row in point_a]
        y = [row[1] for row in point_a]

    ang_a = np.arctan2(y, x)
    return np.rad2deg(ang_a % (2 * np.pi))


def min_max(min, max, hL, hR):
    min = np.minimum(min, np.min(hL))
    min = np.minimum(min, np.min(hR))
    max = np.maximum(max, np.max(hL))
    max = np.maximum(max, np.max(hR))
    return min, max


# ['L_H_K_H','R_H_K_H','L_S_E_W','R_S_E_W','L_K_A_H','R_K_A_H','L_S_H_K','R_S_H_K','L_H_K_A','R_H_K_A','L_H_S_W','R_H_S_W']
def calc(df, angle):
    if angle == 'lH_lK_lA':
        return get_angle("left hip", "left knee", "left ankle", df)
    elif angle == "lS_rS_rE":
        return get_angle("left shoulder", "right shoulder", "right elbow", df)
    elif angle == 'rH_rK_rA':
        return get_angle("right hip", "right knee", "right ankle", df)
    elif angle == 'rS_rH_rK':
        return get_angle("right shoulder", "right hip", "right knee", df)
    elif angle == 'lS_lH_lK':
        return get_angle("left shoulder", "left hip", "left knee", df)
    elif angle == 'lE_lS_lH':
        return get_angle("left elbow", "left shoulder", "left hip", df)
    elif angle == 'rE_rS_rH':
        return get_angle("right elbow", "right shoulder", "right hip", df)
    elif angle == 'rS_N_rW':
        return get_angle("right shoulder", "nose", "right wrist", df)
    elif angle == 'lS_lE_lW':
        return get_angle("left shoulder", "left elbow", "left wrist", df)
    elif angle == 'rS_rE_rW':
        return get_angle("right shoulder", "right elbow", "right wrist", df)
    elif angle == 'lS_N_rS':
        return get_angle("left shoulder", "nose", "right shoulder", df)
    elif angle == 'lE_lS_N':
        return get_angle("left elbow", "left shoulder", "nose", df)
    else:
        import pdb;
        pdb.set_trace()
    # return rH_rK_rA, rS_N_rW, rS_rH_rK, lS_lH_lK


def single_processor(data):
    X_test = []
    x_range_hka = np.linspace(0, 360, num=73)
    df_test = data.iloc[:, :]
    index_list = [-1] * len(plot_list)
    for id, angle in enumerate(plot_list):
        angle_vals = calc(df_test, angle)
        if angle == 'lH_lK_lA':
            counts, bins = np.histogram(angle_vals, np.arange(195, 359, 5))
        else:
            counts, bins = np.histogram(angle_vals, np.arange(0, 359, 5))  # No thresholding, thrsold later for S_H_K

        index = np.argmax(counts)
        index_list[id] = np.where(x_range_hka == bins[index])[0][0]
    X_test.append(index_list)
    return X_test
