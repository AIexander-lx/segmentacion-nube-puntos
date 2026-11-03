import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from segmentacion.semantic_classifier import CLASS_IDS, RuleConfig
from segmentacion.semantic_v3 import _region_prediction, estimate_peak_memory_bytes, export_v3, reconcile_context_votes, refine_component


class SemanticV3Tests(unittest.TestCase):
    def setUp(self):
        # An elongated, normally coherent synthetic region crosses x=2.0.
        x = np.arange(0.1, 3.1, 0.075)
        self.points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
        self.normals = np.tile([0.0, 0.0, 1.0], (len(x), 1))
        self.colors = np.tile([0.5, 0.5, 0.5], (len(x), 1))
        self.v1 = np.full(len(x), CLASS_IDS["unknown"], dtype=np.int8)
        self.v1[0] = CLASS_IDS["wall"]
        self.mapping = np.arange(len(x), dtype=np.int64)
        self.indices = np.arange(len(x), dtype=np.int64)
        self.instances = np.full(len(x), -1, dtype=np.int32)
        self.instances[0] = 12

    def run_v3(self, **kwargs):
        limits = {
            "tile_size": 2.0, "halo": 0.3, "voxel_size": 0.16,
            "max_tile_points": 200, "max_edges": 10000,
            "max_tiles": 20, "max_context_points": 1000,
        }
        limits.update(kwargs)
        return refine_component(
            self.points, self.normals, self.colors, self.v1, self.v1,
            self.mapping, self.indices, instance_v1_global=self.instances,
            **limits,
        )

    def test_cross_tile_region_gets_proposal_and_preserves_v1(self):
        arrays, report = self.run_v3()
        self.assertEqual(arrays["semantic_global"][0], CLASS_IDS["wall"])
        self.assertEqual(arrays["source_global"][0], 0)
        self.assertEqual(arrays["instance_global"][0], 12)
        self.assertTrue(np.all(arrays["semantic_global"][1:30] == CLASS_IDS["beam"]))
        self.assertTrue(np.all(arrays["semantic_global"][30:] == CLASS_IDS["unknown"]))
        self.assertTrue(np.all(arrays["source_global"][1:30] == 1))
        self.assertEqual(report["counts"]["tiles"], 2)
        self.assertEqual(report["counts"]["v1_known_changed_global"], 0)
        self.assertTrue(np.all(arrays["instance_global"][26:30] == -1))
        np.testing.assert_array_equal(arrays["semantic_original"], arrays["semantic_global"])

    def test_deterministic_labels_and_tile_ids_are_stable(self):
        first, _ = self.run_v3()
        second, _ = self.run_v3()
        for key in ("semantic_global", "instance_global", "source_global", "reason_global"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_invalid_budget_and_mapping_fail_closed(self):
        with self.assertRaises(ValueError):
            self.run_v3(halo=2.0)
        bad_mapping = self.mapping.copy()
        bad_mapping[3] = 999
        with self.assertRaises(ValueError):
            refine_component(
                self.points, self.normals, self.colors, self.v1, self.v1,
                bad_mapping, self.indices, instance_v1_global=self.instances,
                tile_size=2.0, halo=0.2, voxel_size=0.16,
                max_tile_points=200, max_edges=10000,
            )
        with self.assertRaises(MemoryError):
            self.run_v3(max_tile_points=2)

    def test_missing_context_is_not_a_negative_vote_and_conflicts_abstain(self):
        owner = np.array([7, 4, 4, 7], dtype=np.int8)
        votes = np.zeros((4, 8), dtype=np.uint8)
        votes[1, 4] = 2
        votes[2, 5] = 1
        votes[3, 3] = 1
        votes[3, 4] = 1
        final, accepted, conflict = reconcile_context_votes(owner, votes)
        np.testing.assert_array_equal(final, [7, 4, 7, 7])
        np.testing.assert_array_equal(accepted, [False, True, False, False])
        np.testing.assert_array_equal(conflict, [False, False, True, True])

    def test_ambiguous_scattering_region_is_not_forced_to_other(self):
        index = np.arange(100, dtype=float)
        z = 1 - 2 * (index + 0.5) / len(index)
        angle = np.pi * (1 + np.sqrt(5)) * index
        points = np.column_stack((np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z))
        prediction, _ = _region_prediction(
            points, points.copy(), np.full_like(points, 0.5),
            (np.full(3, -1.0), np.ones(3)), 2,
            RuleConfig(), 20,
        )
        self.assertEqual(prediction["class"], "unknown")
        self.assertIn("no_specific_semantic_evidence", prediction["warnings"])

    def test_memory_preflight_is_monotonic_and_validated(self):
        small = estimate_peak_memory_bytes(100, 50, max_tile_points=100, max_edges=100, max_context_points=100)
        large = estimate_peak_memory_bytes(200, 50, max_tile_points=100, max_edges=100, max_context_points=100)
        self.assertGreater(large, small)
        with self.assertRaises(ValueError):
            estimate_peak_memory_bytes(-1, 10, max_tile_points=10, max_edges=10, max_context_points=10)

    def test_ply_export_preserves_xyz_normals_and_all_points(self):
        arrays, report = self.run_v3()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.ply"
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(self.points)
            source.normals = o3d.utility.Vector3dVector(self.normals)
            source.colors = o3d.utility.Vector3dVector(self.colors)
            self.assertTrue(o3d.io.write_point_cloud(str(source_path), source))
            outputs = export_v3(source_path, Path(directory) / "v3", arrays, report)
            exported = o3d.io.read_point_cloud(outputs["ply"])
            self.assertEqual(len(exported.points), len(self.points))
            np.testing.assert_array_equal(np.asarray(exported.points), np.asarray(source.points))
            np.testing.assert_array_equal(np.asarray(exported.normals), np.asarray(source.normals))
            self.assertTrue(Path(outputs["npz"]).is_file())
            self.assertTrue(Path(outputs["figure"]).is_file())


if __name__ == "__main__":
    unittest.main()
