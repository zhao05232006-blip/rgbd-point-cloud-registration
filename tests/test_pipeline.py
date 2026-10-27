import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import align, backproject, reference_transform, score


def make_cloud(points, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    return cloud


class PipelineTests(unittest.TestCase):
    def test_backprojection_coordinates_colors_and_invalid_depths(self):
        depth = np.array([[1000, 0, np.nan, np.inf], [2000, 4000, 3000, -5]])
        color = np.zeros((2, 4, 3), dtype=np.uint8)
        color[0, 0] = [255, 0, 64]
        color[1, 0] = [32, 128, 255]
        K = [[2, 0, 1], [0, 4, 0.5], [0, 0, 1]]

        cloud = backproject(color, depth, K, depth_scale=1000, depth_max=3)

        np.testing.assert_allclose(np.asarray(cloud.points), [[-0.5, -0.125, 1], [-1, 0.25, 2]])
        np.testing.assert_allclose(np.asarray(cloud.colors), np.array([[255, 0, 64], [32, 128, 255]]) / 255)

    def test_backprojection_matches_open3d_with_same_camera_parameters(self):
        color = np.arange(24, dtype=np.uint8).reshape(2, 4, 3) * 10
        depth = np.array([[0, 1000, 1500, 3000], [2000, 2999, 4000, 500]], dtype=np.uint16)
        K = np.array([[510, 0, 1.5], [0, 505, 0.7], [0, 0, 1]])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth),
            depth_scale=1000, depth_trunc=3, convert_rgb_to_intensity=False,
        )
        camera = o3d.camera.PinholeCameraIntrinsic(4, 2, 510, 505, 1.5, 0.7)
        expected = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

        actual = backproject(color, depth, K, depth_scale=1000, depth_max=3)

        np.testing.assert_allclose(np.asarray(actual.points), np.asarray(expected.points), atol=1e-7)
        np.testing.assert_allclose(np.asarray(actual.colors), np.asarray(expected.colors), atol=1e-12)

    def test_reference_transform_maps_source_camera_into_target_camera(self):
        manifest = {"frames": [
            {"world_to_camera": [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]},
            {"world_to_camera": [[0, 0, 1, -2], [0, 1, 0, 0.5], [-1, 0, 0, 4], [0, 0, 0, 1]]},
        ]}
        # World origin and unit axes, written directly in each camera's coordinates.
        source_points = np.array([[1, 2, 3, 1], [1, 3, 3, 1], [0, 2, 3, 1], [1, 2, 4, 1]])
        target_points = np.array([[-2, 0.5, 4, 1], [-2, 0.5, 3, 1], [-2, 1.5, 4, 1], [-1, 0.5, 4, 1]])

        transform = reference_transform(manifest, source_index=0, target_index=1)

        np.testing.assert_allclose((transform @ source_points.T).T, target_points, atol=1e-12)

    def test_score_has_no_rmse_when_there_are_no_correspondences(self):
        source = make_cloud([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        target = make_cloud([[10, 0, 0], [11, 0, 0], [10, 1, 0]])

        result = score(source, target, np.eye(4), threshold=0.03)

        self.assertEqual(result["correspondences"], 0)
        self.assertEqual(result["fitness"], 0.0)
        self.assertIsNone(result["inlier_rmse_m"])

    def test_icp_recovers_small_known_pose_on_nonplanar_cloud(self):
        rng = np.random.default_rng(42)
        faces = []
        normals = []
        for axis in range(3):
            face = rng.uniform(0.15, 1.0, size=(180, 3))
            face[:, axis] = 0
            faces.append(face)
            normals.append(np.tile(np.eye(3)[axis], (len(face), 1)))
        points = np.vstack(faces)
        normals = np.vstack(normals)
        angle = np.deg2rad(3)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        translation = np.array([0.015, -0.01, 0.02])
        expected = np.eye(4)
        expected[:3, :3] = rotation
        expected[:3, 3] = translation
        source = make_cloud(points, normals)
        target = make_cloud(points @ rotation.T + translation, normals @ rotation.T)

        result = align(source, target, threshold=0.12)

        np.testing.assert_allclose(result.transformation, expected, atol=1e-5)
        self.assertAlmostEqual(result.fitness, 1.0)
        self.assertLess(result.inlier_rmse, 1e-5)


if __name__ == "__main__":
    unittest.main()
