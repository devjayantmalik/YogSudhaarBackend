import numpy as np
import pandas as pd
from torch.utils.data import Dataset

DEBUG = False
import statsmodels.api as sm


class LandmarksDataset(Dataset):
    def __init__(self):
        self.coords_ids = [i for i in range(33)]

    def max_min(self, df, select):
        mn = 10.0
        mx = -10.0

        for i in select:
            mn = min(mn, min(df[i]))
            mx = max(mx, max(df[i]))
        return mx, mn

    def _get_pose_size(self, landmarks, torso_size_multiplier=2.5):
        left_hip = landmarks[:, 23 * 3:23 * 3 + 2]
        right_hip = landmarks[:, 24 * 3:24 * 3 + 2]
        hips = (left_hip)

        left_shoulder = landmarks[:, 11 * 3:11 * 3 + 2]
        right_shoulder = landmarks[:, 12 * 3:12 * 3 + 2]
        shoulders = (left_shoulder)

        torso_size = np.linalg.norm(shoulders - hips, axis=1)
        pose_center = hips
        total_joints = [i for i in range(99)]
        norm_vals = landmarks[:, np.ravel([total_joints[i * 3:i * 3 + 2] for i in self.coords_ids])] - np.tile(
            pose_center, len(self.coords_ids))
        max_dist = np.max(np.linalg.norm(norm_vals.reshape(len(norm_vals), -1, 2), axis=2), axis=1)
        return np.maximum(torso_size * torso_size_multiplier, max_dist)

    def smooth(self, df, select, norm=True, f=0):
        new_df = pd.DataFrame()
        if f == 0:
            f = min(0.03, (41 / len(df)))
        time = [i for i in range(len(df))]

        all = []
        for i in range(33):
            all.append('x_' + str(i))
            all.append('y_' + str(i))
            all.append('z_' + str(i))

        for i in all:
            new_df[i] = sm.nonparametric.lowess(df[i].values, time, frac=f,
                                                it=3, delta=0.0, is_sorted=True,
                                                missing='drop', return_sorted=False)
        joint_ids = self.coords_ids
        skel = new_df.to_numpy()

        joints_data = np.zeros((len(skel), len(joint_ids) * 3))
        for index, id in enumerate(joint_ids):
            joints_data[:, index * 3:index * 3 + 3] = skel[:, id * 3:id * 3 + 3] - (skel[:, 23 * 3:23 * 3 + 3])
        norm0, norm1, norm2 = False, True, False
        if norm0:
            x_max, x_min = self.max_min(new_df, all[1::3])
            y_max, y_min = self.max_min(new_df, all[2::3])
            z_max, z_min = self.max_min(new_df, all[3::3])

            for i in select[0::3]:
                new_df[i] = (new_df[i] - x_min) / (x_max - x_min)

            for i in select[1::3]:
                new_df[i] = 1 - ((new_df[i] - y_min) / (y_max - y_min))

            for i in select[2::3]:
                new_df[i] = (new_df[i] - z_min) / (z_max - z_min)
            normalised_joints3 = new_df.to_numpy()
        elif norm1:

            if DEBUG:
                print("file len", len(joints_data))
            shifted_joints = joints_data

            x_min_data, x_max_data = shifted_joints[:, 0::3].min(axis=1), shifted_joints[:, 0::3].max(axis=1)
            y_min_data, y_max_data = shifted_joints[:, 1::3].min(axis=1), shifted_joints[:, 1::3].max(axis=1)
            z_min_data, z_max_data = shifted_joints[:, 2::3].min(axis=1), shifted_joints[:, 2::3].max(axis=1)

            normalised_joints3 = shifted_joints.copy()
            normalised_joints3[:, 0::3] = (normalised_joints3[:, 0::3] - x_min_data[:, None]) / (
                    x_max_data[:, None] - x_min_data[:, None])
            normalised_joints3[:, 1::3] = (normalised_joints3[:, 1::3] - y_min_data[:, None]) / (
                    y_max_data[:, None] - y_min_data[:, None])
            normalised_joints3[:, 2::3] = (normalised_joints3[:, 2::3] - z_min_data[:, None]) / (
                    z_max_data[:, None] - z_min_data[:, None])
            if 0 in (x_max_data[:, None] - x_min_data[:, None]) or 0 in (
                    y_max_data[:, None] - y_min_data[:, None]) or 0 in (z_max_data[:, None] - z_min_data[:, None]):
                print("null in data")

        elif norm2:
            body_size = self._get_pose_size(skel)
            joints_data /= np.repeat(body_size.reshape(-1, 1), joints_data.shape[1], axis=1)
            normalised_joints3 = joints_data
        else:
            normalised_joints3 = joints_data

        column_labels = []
        for id in joint_ids:
            column_labels.extend(['x' + str(id), 'y' + str(id), 'z' + str(id)])

        new_df = pd.DataFrame(normalised_joints3, columns=column_labels)
        '''x_max, x_min = max_min(new_df,select[0::3])
        y_max, y_min = max_min(new_df,select[1::3])
        z_max, z_min = max_min(new_df,select[2::3])

        print("x = {} : {}\ny = {} : {}\nz = {} : {}".format(x_min,x_max,y_min,y_max,z_min,z_max))'''

        return new_df

    def shift_cols(self, df):
        right_knee = '26'
        left_knee = '25'
        right_heel = '30'
        left_heel = '29'
        right_toe = '32'
        left_toe = '31'

        if sum(df.iloc[:10]['x' + left_knee]) < sum(df.iloc[:10]['x' + right_knee]):
            front_knee = right_knee
            back_knee = left_knee
            front_heel = right_heel
            back_toe = left_toe
        else:
            front_knee = left_knee
            back_knee = right_knee
            front_heel = left_heel
            back_toe = right_toe

        if df['x' + right_knee].isin([-1]).any().any() or df['x' + left_knee].isin([-1]).any().any():
            print("BOTH LEFT AND RIGHT knee has missing data")

        df_org = df.copy()
        coords_id = self.coords_ids
        target_id = [front_knee, back_knee, front_heel, back_toe]
        for id, val in enumerate(coords_id[2:6]):
            mask = ['x' + str(val), 'y' + str(val), 'z' + str(val)]
            target = ['x' + str(target_id[id]), 'y' + str(target_id[id]), 'z' + str(target_id[id])]
            df_org[mask] = df[target]
        return df_org

    def pre_process(self, df):
        select = []
        for coords in range(33):
            select.append('x_' + str(coords))
            select.append('y_' + str(coords))
            select.append('z_' + str(coords))

        df = df[select]

        coords_id = self.coords_ids
        select = []
        for coords in coords_id:
            select.append('x_' + str(coords))
            select.append('y_' + str(coords))
            select.append('z_' + str(coords))

        if len(df) == 0:
            return df
        if df[select[0:-6:3]].isin([-1]).any().any():
            print(df[select[0:-6:3]].isin([-1]).any())
            return pd.DataFrame()

        df = self.smooth(df, select)

        for i in range(0, len(df.columns), 3):
            df.loc[:, 'd' + df.columns[i]] = (df[df.columns[i]] - df[df.columns[i]].shift(1)).fillna(0)
            df.loc[:, 'd' + df.columns[i + 1]] = (df[df.columns[i + 1]] - df[df.columns[i + 1]].shift(1)).fillna(0)
            df.loc[:, 'd' + df.columns[i + 2]] = (df[df.columns[i + 1]] - df[df.columns[i + 1]].shift(1)).fillna(0)

        select = []
        for coords in coords_id:
            select.append('x' + str(coords))
            select.append('y' + str(coords))
            select.append('dx' + str(coords))
            select.append('dy' + str(coords))

        df = df[select]
        org_cols = len(df.columns)
        df_copy = df.copy()
        df_copy = df_copy.iloc[1:, :]
        return df_copy

    def process_data_from_excel(self, Data):
        df = Data
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)

        df = self.pre_process(df)
        df = df.reset_index(drop=True)

        return df

    def add_staionary_points(self, df):
        assert len(df.columns) == 100, 'Mediapipe not used for pose estimation ' + str(len(df.columns))
        top_id = (len(df.columns) - 1) // 3
        bottom_id = top_id + 1
        col_index = np.repeat([top_id, bottom_id], 3)
        vals = [.5, 0, 0, .5, 1, 0]
        for i, j, k in zip(['x', 'y', 'z', 'x', 'y', 'z'], col_index, vals):
            df.loc[:, str(i) + str(j)] = k

        return df

    def __len__(self, ):
        return len(self.landmarks_frame_map)

    def __getitem__(self, idx, ):
        self.landmarks_frame = self.landmarks_frame_map[str(idx.tolist()[0])]
        batch_size = len(self.landmarks_frame.iloc[:, 0])
        idx = np.arange(0, batch_size)

        idx = [i for i in idx if i < len(self.landmarks_frame) - 1]

        out_idx = [1 + np.random.randint(i, min(i + self.future_offset, len(idx))) for i in range(len(idx))]
        if self.test:
            print(idx, "testing", out_idx)
            self.future_offset = 1
            out_idx = [min(i + self.future_offset, len(idx)) for i in range(len(idx))]

        time = np.array([i for i in range(len(idx))])
        landmarks = self.landmarks_frame.iloc[idx, :]
        landmarks = np.array(landmarks)
        output_landmarks = self.landmarks_frame.iloc[out_idx, :]
        output_landmarks = np.array(output_landmarks)
        sample = {"time": time, 'landmarks': landmarks, "output": output_landmarks,
                  "col_id": self.landmarks_frame.columns[:]}

        if self.transform:
            sample = self.transform(sample)
        return sample
