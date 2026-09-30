"""Hard pairwise topology constraints for contact-anchor optimization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

MAX_RELATIVE_DIRECTION_DEGREES = 45.0
MIN_PAIR_DISTANCE_RATIO = 1.0
DISTANCE_FEASIBILITY_MARGIN_M = 1.0e-6
GEOMETRY_EPSILON_M = 1.0e-12
CONTACTS_PER_PAIR = 2


@dataclass(frozen=True)
class PairwiseTopologyConstraint:
    pair_ids: np.ndarray
    reference_vectors: np.ndarray
    reference_distances: np.ndarray
    minimum_cosine: float

    @classmethod
    def from_reference(
        cls,
        reference: np.ndarray,
        active: np.ndarray,
        *,
        max_direction_degrees: float,
    ) -> "PairwiseTopologyConstraint":
        columns = np.flatnonzero(active)
        pair_ids = np.asarray(
            tuple(combinations(columns.tolist(), CONTACTS_PER_PAIR)), dtype=np.int32,
        ).reshape(-1, CONTACTS_PER_PAIR)
        vectors = reference[pair_ids[:, 0]] - reference[pair_ids[:, 1]]
        distances = np.linalg.norm(vectors, axis=1)
        if np.any(distances <= GEOMETRY_EPSILON_M):
            raise ValueError("Original stable contacts contain a degenerate pair")
        angle = np.deg2rad(float(max_direction_degrees))
        if not 0.0 <= angle < np.pi:
            raise ValueError("Relative-direction limit must lie in [0, 180) degrees")
        return cls(pair_ids, vectors, distances, float(np.cos(angle)))

    def accept(self, positions: np.ndarray) -> np.ndarray:
        candidates = np.asarray(positions, dtype=np.float64)
        if candidates.ndim != 3 or candidates.shape[2] != 3:
            raise ValueError("Contact proposals must have shape (B, K, 3)")
        vectors = candidates[:, self.pair_ids[:, 0]] - candidates[:, self.pair_ids[:, 1]]
        distances = np.linalg.norm(vectors, axis=2)
        dot = np.einsum("bpi,pi->bp", vectors, self.reference_vectors)
        denominator = distances * self.reference_distances[None]
        cosine = np.divide(
            dot,
            denominator,
            out=np.full_like(dot, -np.inf),
            where=denominator > GEOMETRY_EPSILON_M,
        )
        direction_ok = (distances > GEOMETRY_EPSILON_M) & (cosine >= self.minimum_cosine)
        minimum = MIN_PAIR_DISTANCE_RATIO * self.reference_distances[None]
        distance_ok = distances >= np.nextafter(minimum, -np.inf)
        return np.all(direction_ok & distance_ok, axis=1)

    def report(self, positions: np.ndarray) -> dict:
        positions = np.asarray(positions, dtype=np.float64)
        vectors = positions[self.pair_ids[:, 0]] - positions[self.pair_ids[:, 1]]
        distances = np.linalg.norm(vectors, axis=1)
        dot = np.einsum("pi,pi->p", vectors, self.reference_vectors)
        denominator = distances * self.reference_distances
        cosine = np.divide(
            dot,
            denominator,
            out=np.full_like(dot, -1.0),
            where=denominator > GEOMETRY_EPSILON_M,
        )
        angles = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return {
            "pair_ids": self.pair_ids.tolist(),
            "direction_change_degrees": angles.tolist(),
            "distance_ratio": (distances / self.reference_distances).tolist(),
            "all_constraints_satisfied": bool(self.accept(positions[None])[0]),
        }
