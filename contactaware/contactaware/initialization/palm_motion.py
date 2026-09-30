"""Geodesic blending of palm poses."""


from scipy.spatial.transform import Rotation


def blended_pose(reference, carried, weight):
    """Geodesic step from the demonstrated pose toward the carried one."""
    position = reference[0] + weight * (carried[0] - reference[0])
    delta = Rotation.from_matrix(reference[1].T @ carried[1]).as_rotvec()
    return position, reference[1] @ Rotation.from_rotvec(weight * delta).as_matrix()


