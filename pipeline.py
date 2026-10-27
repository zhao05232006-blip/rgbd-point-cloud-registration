"""Small functions for RGB-D back-projection, cleaning and pairwise ICP."""
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

DATA = Path(__file__).resolve().parent / "data" / "fountain"


def backproject(color, depth, K, depth_scale=1000.0, depth_max=3.0):
    """Return a colored cloud. Camera coordinates: x right, y down, z forward."""
    if depth.ndim != 2 or color.shape != (*depth.shape, 3):
        raise ValueError("Expected aligned HxW depth and HxWx3 RGB images")
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K must have positive focal lengths")
    if depth_scale <= 0 or depth_max <= 0:
        raise ValueError("Depth scale and maximum depth must be positive")
    z = depth.astype(np.float64) / depth_scale
    v, u = np.indices(depth.shape)
    valid = np.isfinite(z) & (z > 0) & (z < depth_max)
    # Pinhole model: one pixel + its depth gives one 3D point.
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.column_stack((x[valid], y[valid], z[valid])))
    cloud.colors = o3d.utility.Vector3dVector(color[valid].astype(float) / 255.0)
    return cloud


def load_frame(index, keep_columns=None):
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    frame = manifest["frames"][index]
    color = np.asarray(o3d.io.read_image(str(DATA / frame["color"]))).copy()
    depth = np.asarray(o3d.io.read_image(str(DATA / frame["depth"]))).copy()
    if keep_columns is not None:
        # Mask rather than resize: the principal point must stay unchanged.
        start, stop = keep_columns
        depth[:, :start] = 0
        depth[:, stop:] = 0
    cloud = backproject(color, depth, manifest["K"], manifest["depth_scale"], manifest["depth_max_m"])
    return cloud, color, depth, manifest


def preprocess(cloud, voxel_size):
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive")
    down = cloud.voxel_down_sample(voxel_size)
    if len(down.points) < 30:
        raise ValueError("Too few valid points for filtering and normal estimation")
    clean, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel_size, max_nn=30))
    return clean, {"raw": len(cloud.points), "voxel": len(down.points), "filtered": len(clean.points)}


def align(source, target, threshold, initial=None, max_iterations=80):
    if threshold <= 0:
        raise ValueError("Correspondence threshold must be positive")
    if initial is None:
        initial = np.eye(4)
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )


def score(source, target, transformation, threshold=0.03):
    result = o3d.pipelines.registration.evaluate_registration(source, target, threshold, transformation)
    count = len(result.correspondence_set)
    return {
        "fitness": float(result.fitness),
        # Open3D returns 0 for empty matches; None avoids presenting that as perfect alignment.
        "inlier_rmse_m": float(result.inlier_rmse) if count else None,
        "correspondences": count,
    }


def reference_transform(manifest, source_index, target_index):
    Es = np.asarray(manifest["frames"][source_index]["world_to_camera"])
    Et = np.asarray(manifest["frames"][target_index]["world_to_camera"])
    return Et @ np.linalg.inv(Es)


def pose_difference(estimate, reference):
    """Consistency with the dataset pose, not an independent accuracy measurement."""
    cosine = (np.trace(estimate[:3, :3] @ reference[:3, :3].T) - 1.0) / 2.0
    return {
        "reference_rotation_difference_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        "reference_translation_difference_m": float(np.linalg.norm(estimate[:3, 3] - reference[:3, 3])),
    }


def transformed(cloud, matrix):
    return copy.deepcopy(cloud).transform(matrix)
