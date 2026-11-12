import unittest

import numpy as np

from segmentacion.local_regions import local_voxel_regions


class LocalRegionTests(unittest.TestCase):
    def test_connected_surfaces_with_normal_transition_split(self):
        grid = np.linspace(0.0, 0.8, 17)
        u, v = np.meshgrid(grid, grid)
        horizontal = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
        vertical = np.column_stack((u.ravel(), np.zeros(u.size), v.ravel()))
        points = np.vstack((horizontal, vertical))
        normals = np.vstack((
            np.tile([0.0, 0.0, 1.0], (len(horizontal), 1)),
            np.tile([0.0, 1.0, 0.0], (len(vertical), 1)),
        ))
        labels, metrics, _ = local_voxel_regions(
            points, normals, voxel_size=0.08, normal_angle_degrees=25,
            min_normal_coherence=0.6, allow_invalid_normals=False,
        )
        horizontal_middle = 5 * len(grid) + 5
        vertical_middle = len(horizontal) + horizontal_middle
        self.assertNotEqual(labels[horizontal_middle], labels[vertical_middle])
        self.assertGreaterEqual(metrics["regions"], 2)
        self.assertEqual(len(labels), len(points))

    def test_curved_surface_connects_with_progressive_normals(self):
        angle, height = np.meshgrid(np.linspace(0, np.pi / 2, 80), np.linspace(0, 0.6, 8))
        points = np.column_stack((np.cos(angle.ravel()), np.sin(angle.ravel()), height.ravel()))
        normals = np.column_stack((np.cos(angle.ravel()), np.sin(angle.ravel()), np.zeros(angle.size)))
        labels, metrics, _ = local_voxel_regions(
            points, normals, voxel_size=0.12, normal_angle_degrees=25,
            max_centroid_distance=0.28, min_region_points=1,
        )
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        largest_fraction = counts.max() / len(points)
        self.assertGreater(largest_fraction, 0.75)
        self.assertGreater(metrics["normal_coverage_percent"], 99.0)

    def test_invalid_normals_and_degenerate_cloud(self):
        points = np.zeros((12, 3))
        normals = np.zeros_like(points)
        labels, metrics, _ = local_voxel_regions(points, normals, voxel_size=0.1)
        self.assertEqual(len(labels), len(points))
        self.assertEqual(metrics["voxel_count"], 1)
        self.assertEqual(metrics["normal_coverage_percent"], 0.0)
        self.assertTrue(np.all(labels == 0))

    def test_determinism_and_index_correspondence(self):
        points = np.array([[0.01, 0, 0], [0.03, 0, 0], [0.25, 0, 0], [0.27, 0, 0]])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        first, _, _ = local_voxel_regions(points, normals, voxel_size=0.1)
        second, _, _ = local_voxel_regions(points, normals, voxel_size=0.1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (len(points),))
        point_global_indices = np.array([41, 42, 93, 95])
        original_to_global = np.array([41, 42, 93, 95, 100])
        global_to_local = {int(value): index for index, value in enumerate(point_global_indices)}
        original_region = np.array([first[global_to_local.get(int(value), 0)] if value in global_to_local else -1
                                    for value in original_to_global])
        np.testing.assert_array_equal(original_region, [first[0], first[1], first[2], first[3], -1])

    def test_explicit_point_and_edge_limits_and_bad_parameters(self):
        points = np.zeros((5, 3))
        normals = np.tile([0.0, 0.0, 1.0], (5, 1))
        with self.assertRaises(MemoryError):
            local_voxel_regions(points, normals, max_points=4)
        with self.assertRaises(MemoryError):
            local_voxel_regions(points, normals, voxel_size=0.01, max_edges=1)
        with self.assertRaises(ValueError):
            local_voxel_regions(points, normals, voxel_size=0)


if __name__ == "__main__":
    unittest.main()
