"""Shared task-aware time segmentation for prediction entry points."""
import numpy as np
import pandas as pd

CADENCE_SECONDS = 2


def contiguous_segments(frame: pd.DataFrame, time_column: str):
    timestamps = pd.to_datetime(frame[time_column], utc=True)
    if timestamps.isna().any():
        raise ValueError('Invalid prediction timestamps')
    identity = [k for k in ('device_id', 'job_id', 'segment_id') if k in frame]
    if identity and frame[identity].isna().any().any():
        raise ValueError('Missing device/task/segment identity')
    groups = frame.groupby(identity, sort=False, dropna=False).indices.values() if identity else [np.arange(len(frame))]
    segments = []
    for positions in groups:
        positions = np.asarray(positions)
        positions = positions[np.argsort(timestamps.iloc[positions].to_numpy(), kind='stable')]
        part = frame.iloc[positions].reset_index(drop=True)
        times = timestamps.iloc[positions].reset_index(drop=True)
        keep = ~times.duplicated(keep='last')
        part, times = (part.loc[keep].reset_index(drop=True), times.loc[keep].reset_index(drop=True))
        labels = times.diff().dt.total_seconds().ne(CADENCE_SECONDS).cumsum()
        for indices in part.groupby(labels, sort=False).groups.values():
            segments.append((part.loc[indices].reset_index(drop=True), times.loc[indices].reset_index(drop=True)))
    yield from sorted(segments, key=lambda pair: pair[1].iloc[-1])
