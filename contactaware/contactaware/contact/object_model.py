"""ContactAware adapter: simulator collision queries and coordinate transforms only."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from contactaware.settings import COMFREE_WARP_ROOT
from contactaware.types import ObjectModel
if str(COMFREE_WARP_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFREE_WARP_ROOT))
from comfree_warp.geometry import ObjectCollisionQuery as ObjectQuery
from comfree_warp.collision_config import CollisionConfig

OBJECT_BODY = "obj"

def load_object_model(xml_path: Path, *, topk: int, distance_offset: float) -> ObjectModel:
    xml_path = Path(xml_path).resolve()
    query = ObjectQuery(xml_path, config=CollisionConfig(topk, distance_offset))
    return ObjectModel(query.support_spheres, xml_path, float(distance_offset), query)


def object_rotation(pose: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(pose[:4]).as_matrix()


def obj_to_world(points_obj: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points_obj @ object_rotation(pose).T + pose[4:7]


def world_to_obj(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points_world - pose[4:7]) @ object_rotation(pose)


def object_query_world(points_world: np.ndarray, pose: np.ndarray,
                       obj: ObjectModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi, surface_obj, normal_obj = object_query_obj(world_to_obj(points_world, pose), obj)
    return phi, obj_to_world(surface_obj, pose), normal_obj @ object_rotation(pose).T


def object_distance_world(points_world: np.ndarray, pose: np.ndarray,
                          obj: ObjectModel) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance and its world-position derivative, without normalizing it.

    A fused unit normal is a contact orientation, not the gradient of the fused
    distance: the latter also differentiates the position-dependent weights.
    Keep both quantities distinct, including their signs (outward vs inward).
    """
    features = obj.query_fn.distance_features(world_to_obj(points_world, pose))
    return features[:, 0], features[:, 1:4] @ object_rotation(pose).T


def object_distance_obj(points_obj: np.ndarray, obj: ObjectModel) -> tuple[np.ndarray, np.ndarray]:
    """Distance and its unnormalized object-position derivative."""
    features = obj.query_fn.distance_features(points_obj)
    return features[:, 0], features[:, 1:4]


def object_query_obj(points_obj: np.ndarray,
                     obj: ObjectModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed distance, surface point and inward normal in the object frame."""
    return obj.query_fn(np.asarray(points_obj, dtype=np.float64))


def object_center(xml_path: Path) -> np.ndarray:
    """Object centre of mass in the object frame."""
    root = ET.parse(xml_path).getroot()
    inertial = root.find(f".//body[@name='{OBJECT_BODY}']/inertial")
    if inertial is not None:
        return np.asarray(inertial.attrib.get("pos", "0 0 0").split(), dtype=np.float32)
    raise ValueError(f"Object body {OBJECT_BODY!r} needs an explicit <inertial> in {xml_path}")
