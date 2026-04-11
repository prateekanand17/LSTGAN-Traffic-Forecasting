import os
import numpy as np
import pandas as pd
import torch
from src.config import *

def get_geographic_coordinates(sensor_ids):
    fallback_lats = dict(zip(sensor_ids, np.random.uniform(37.25, 37.43, len(sensor_ids))))
    fallback_lngs = dict(zip(sensor_ids, np.random.uniform(-122.08, -121.84, len(sensor_ids))))
    
    if os.path.exists(STATIONS_CSV):
        df = pd.read_csv(STATIONS_CSV)
        if 'sensor_id' in df.columns:
            df['sensor_id'] = df['sensor_id'].astype(str)
            mapping = df.set_index('sensor_id')
            
            real_lats = []
            real_lngs = []
            for sid in sensor_ids:
                if sid in mapping.index:
                    real_lats.append(mapping.loc[sid, 'latitude'])
                    real_lngs.append(mapping.loc[sid, 'longitude'])
                else:
                    real_lats.append(fallback_lats[sid])
                    real_lngs.append(fallback_lngs[sid])
            return np.array(real_lats), np.array(real_lngs)
    return np.array(list(fallback_lats.values())), np.array(list(fallback_lngs.values()))

class TrafficDatasetLite:
    def __init__(self, data_df, mean, std):
        self.num_sensors = data_df.shape[1]
        self.mean = mean; self.std = std
        self.speed = ((data_df.values - mean) / (std + 1e-8)).astype(np.float32)
        if hasattr(data_df.index, 'weekday'):
            self.day_of_week = data_df.index.weekday.values
            self.time_of_day = (data_df.index.hour * 12 + data_df.index.minute // 5).values
        else:
            self.day_of_week = np.arange(len(data_df)) % 7
            self.time_of_day = np.arange(len(data_df)) % 288

    def _time_feats(self, idx, length):
        s = idx - length
        dow = self.day_of_week[s:idx].astype(np.float32) / 6.0
        tod = self.time_of_day[s:idx].astype(np.float32) / 287.0
        return np.stack([dow, tod], axis=-1)

    def get_sample(self, t):
        N = self.num_sensors
        sp_w = self.speed[t - WEEKLY_WINDOW:t]; tf_w = self._time_feats(t, WEEKLY_WINDOW)
        X_w = np.stack([sp_w, np.tile(tf_w[:, 0:1], (1, N)), np.tile(tf_w[:, 1:2], (1, N))], axis=-1)
        sp_d = self.speed[t - DAILY_WINDOW:t]; tf_d = self._time_feats(t, DAILY_WINDOW)
        X_d = np.stack([sp_d, np.tile(tf_d[:, 0:1], (1, N)), np.tile(tf_d[:, 1:2], (1, N))], axis=-1)
        X_h = self.speed[t - HOURLY_WINDOW:t][:, :, np.newaxis]
        t_info = np.array([self.day_of_week[t], self.time_of_day[t]], dtype=np.int64)
        Y = self.speed[t:t + FORECAST_HORIZON][:, :, np.newaxis]
        return [torch.from_numpy(x).unsqueeze(0).to(DEVICE) for x in (X_w, X_d, X_h, t_info, Y)]

    def find_matching_index(self, day_of_week, hour):
        tod = hour * STEPS_PER_HOUR
        for t in range(WEEKLY_WINDOW, len(self.speed) - FORECAST_HORIZON):
            if self.day_of_week[t] == day_of_week and self.time_of_day[t] == tod:
                return t
        return None
