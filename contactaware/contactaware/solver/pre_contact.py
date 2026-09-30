"""Hand-object clearance required before contact guidance starts."""

import numpy as np


class PreContactClearance:
    """Per-query clearance at a source frame.

    A query keeps ``distance`` before its finger's first stable contact; the palm and fingers without stable
    contacts use the hand's first contact. Afterwards the clearance is ``base``. The clearance never increases
    in time, so a swept interval uses the clearance of its later frame.
    """

    def __init__(self, hand, anchors, distance, base):
        mask = np.asarray(anchors.mask, dtype=bool)
        starts = {}
        for column, finger in enumerate(np.asarray(anchors.finger_ids, dtype=np.int64)):
            first = int(np.flatnonzero(mask[:, column])[0])
            starts[int(finger)] = min(first, starts.get(int(finger), first))
        hand_start = min(starts.values())
        fingers = np.asarray(hand.query_points.finger_ids, dtype=np.int64)
        self.onset = np.asarray([starts.get(int(f), hand_start) for f in fingers], dtype=np.int64)
        self.base = float(base)
        self.maximum = max(float(distance), self.base)

    def __call__(self, frame, queries=None):
        onset = self.onset if queries is None else self.onset[np.asarray(queries, dtype=np.int64)]
        return np.where(int(frame) < onset, self.maximum, self.base)
