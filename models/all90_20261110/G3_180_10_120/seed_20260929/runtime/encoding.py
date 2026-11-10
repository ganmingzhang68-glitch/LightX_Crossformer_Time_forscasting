import hashlib
import json
import numpy as np
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

class Encoder:

    def __init__(self, c, pre):
        if pre['contract_sha'] != digest(c):
            raise ValueError('Encoder contract mismatch')
        self.c, self.pre = (c, pre)
        self.mean, self.scale = (np.asarray(pre['mean'], dtype='float32'), np.asarray(pre['scale'], dtype='float32'))
        if self.mean.shape != (218,) or self.scale.shape != (218,) or (not np.isfinite(self.scale).all()) or (self.scale <= 0).any():
            raise ValueError('Invalid scaler')
        self.dynamic_indices = list(range(128)) + [c['source_features'].index(f) for f in c['dynamic_added']]

    def encode(self, raw, quality):
        raw = np.asarray(raw, dtype='float32')
        known = np.isfinite(raw)
        x = np.where(known, (raw - self.mean) / self.scale, 0.0).astype('float32')
        conditions = x[:, 128:].copy()
        for f in self.c['categorical']:
            j = self.c['added'].index(f)
            v = raw[:, 128 + j]
            tokens = np.where(known[:, 128 + j], 1, 0).astype('float32')
            for token, value in enumerate(self.pre['categorical_vocabulary'][f], 2):
                tokens[v == value] = token
            conditions[:, j] = tokens
        aux = np.concatenate([known[:, 128:].astype('float32'), np.asarray(quality, dtype='float32')], axis=1)
        dynamic = x[:, self.dynamic_indices]
        if not all((np.isfinite(v).all() for v in (dynamic, conditions, aux))):
            raise ValueError('Invalid encoded input')
        return (dynamic, conditions, aux)
