import unittest

import numpy as np

from segmentacion.hybrid_pipeline import (
    SOURCE_IDS,
    assign_tile_keys,
    build_spatial_tile_plan,
    compare_tile_proposals,
    finalize_tiled_predictions,
    fuse_unknown_only,
    make_semantic_cloud,
    preflight_tile_plan,
    project_roi_to_original,
    proposal_rejection_flags,
    select_roi_indices,
    tile_context_indices,
)
from segmentacion.semantic_classifier import CLASS_IDS, semantic_colors


class HybridPipelineTests(unittest.TestCase):
    def test_consensus_unknown_votes_abstentions_and_contradictions(self):
        v1 = np.array([7, 7, 7, 2, 7], dtype=np.int8)
        owner = np.array([5, 5, 5, 4, 7], dtype=np.int8)
        votes = np.zeros((5, len(CLASS_IDS)), dtype=np.uint16)
        votes[0, 5], votes[0, 7] = 2, 1
        votes[1, 5], votes[1, 4] = 2, 1
        votes[2, 5] = 1
        votes[3, 4] = 2
        votes[4, 7], votes[4, 5] = 1, 2
        args = (v1, owner, np.zeros(5, dtype=np.int32), np.ones(5, dtype=np.uint8), votes)
        final, _, metrics = finalize_tiled_predictions(*args, mode="consensus")
        np.testing.assert_array_equal(final, [5, 7, 7, 2, 7])
        np.testing.assert_array_equal(final, finalize_tiled_predictions(*args, mode="consensus")[0])
        self.assertEqual(metrics["consensus_abstentions"], 3)
        with self.assertRaises(ValueError):
            finalize_tiled_predictions(*args, mode="consensus", min_known_votes=1)
        protected, _, _ = finalize_tiled_predictions(*args, mode="consensus", mapping_valid=np.array([True, True, True, False, True]))
        self.assertEqual(protected[3], 2)

    def test_shifted_partition_preserves_indices(self):
        points = np.array([[0.2, 0, 0], [0.9, 0, 0], [1.2, 0, 0]])
        a = build_spatial_tile_plan(points, tile_size=1.0, halo=0.2, origin=np.zeros(3))
        b = build_spatial_tile_plan(points, tile_size=1.0, halo=0.2, origin=np.array([0.5, 0, 0]))
        self.assertFalse(np.array_equal(a.point_tile_ids, b.point_tile_ids))
        np.testing.assert_array_equal(np.sort(b.sorted_point_indices), np.arange(len(points)))
    def test_roi_is_half_open_and_has_stable_global_rows(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
        ])
        rows = select_roi_indices(points, (np.zeros(3), np.ones(3)), max_points=10)
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(points[rows], points[:2])

    def test_unknown_only_fusion_preserves_v1_and_records_sources(self):
        v1 = np.array([0, 7, 7, 3], dtype=np.int8)
        proposals = np.array([2, 4, 7, 5], dtype=np.int8)
        regions = np.array([0, 2, 3, 4], dtype=np.int32)
        v2, source, metrics = fuse_unknown_only(v1, proposals, regions)
        np.testing.assert_array_equal(v2, [0, 4, 7, 3])
        np.testing.assert_array_equal(source, [
            SOURCE_IDS["v1"], SOURCE_IDS["local_region"],
            SOURCE_IDS["unknown"], SOURCE_IDS["v1"],
        ])
        self.assertEqual(metrics["known_v1_preserved"], 2)
        self.assertEqual(metrics["proposals_ignored_on_known_v1"], 2)
        self.assertEqual(metrics["local_proposals_applied"], 1)
        np.testing.assert_array_equal(v1, [0, 7, 7, 3])

    def test_mapping_errors_and_invalid_proposals_stay_unknown(self):
        v1 = np.array([7, 2, 7], dtype=np.int8)
        proposal = np.array([9, 0, 4], dtype=np.int16)
        region = np.array([0, 1, -1], dtype=np.int32)
        mapping_valid = np.array([True, True, False])
        v2, source, metrics = fuse_unknown_only(v1, proposal, region, mapping_valid=mapping_valid)
        np.testing.assert_array_equal(v2, [7, 2, 7])
        self.assertEqual(source[1], SOURCE_IDS["v1"])
        self.assertEqual(source[2], SOURCE_IDS["unknown"])
        self.assertEqual(metrics["invalid_mapping_rows"], 1)
        self.assertEqual(metrics["invalid_proposal_rows"], 1)

    def test_original_projection_is_roi_limited_and_checks_alignment(self):
        global_v1 = np.array([7, 2, 7, 4], dtype=np.int8)
        original_to_global = np.array([1, 2, 2, 3, -1])
        original_v1 = np.array([2, 7, 7, 4, 7], dtype=np.int8)
        roi_indices = np.array([1, 2], dtype=np.int64)
        hybrid = np.array([2, 3], dtype=np.int8)
        sources = np.array([SOURCE_IDS["v1"], SOURCE_IDS["local_region"]], dtype=np.uint8)
        result = project_roi_to_original(
            roi_global_indices=roi_indices,
            hybrid_roi_labels=hybrid,
            hybrid_roi_sources=sources,
            v1_global_labels=global_v1,
            v1_original_labels=original_v1,
            original_to_global=original_to_global,
        )
        np.testing.assert_array_equal(result["original_indices"], [0, 1, 2])
        np.testing.assert_array_equal(result["original_hybrid_labels"], [2, 3, 3])
        self.assertEqual(result["mapping_inconsistent_rows"], 0)

        inconsistent_v1 = original_v1.copy()
        inconsistent_v1[2] = 1
        bad = project_roi_to_original(
            roi_global_indices=roi_indices,
            hybrid_roi_labels=hybrid,
            hybrid_roi_sources=sources,
            v1_global_labels=global_v1,
            v1_original_labels=inconsistent_v1,
            original_to_global=original_to_global,
        )
        self.assertEqual(bad["original_hybrid_labels"][2], CLASS_IDS["unknown"])
        self.assertEqual(bad["mapping_inconsistent_rows"], 1)

    def test_tile_boundary_ownership_is_half_open_and_deterministic(self):
        points = np.array([[0.99, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        first = assign_tile_keys(points, np.zeros(3), 1.0)
        second = assign_tile_keys(points, np.zeros(3), 1.0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 0], [0, 1, 2])

    def test_halo_adds_context_without_changing_unique_owner(self):
        points = np.array([[0.95, 0.0, 0.0], [1.05, 0.0, 0.0]])
        with_halo = build_spatial_tile_plan(points, tile_size=1.0, halo=0.2, origin=np.zeros(3))
        without_halo = build_spatial_tile_plan(points, tile_size=1.0, halo=0.0, origin=np.zeros(3))
        np.testing.assert_array_equal(with_halo.point_tile_ids, [0, 1])
        self.assertEqual(len(tile_context_indices(without_halo, points, 0)), 1)
        self.assertEqual(len(tile_context_indices(with_halo, points, 0)), 2)
        np.testing.assert_array_equal(with_halo.owner_indices(0), [0])
        np.testing.assert_array_equal(with_halo.owner_indices(1), [1])

    def test_class_and_known_unknown_conflicts_are_separate_from_instance_ids(self):
        owner = np.array([5, 7, 4, 7], dtype=np.int8)
        ownership = np.ones(4, dtype=np.uint8)
        votes = np.zeros((4, len(CLASS_IDS)), dtype=np.uint16)
        votes[0, 5] = 1
        votes[0, 4] = 1  # Known class vs another known class.
        votes[1, 7] = 1
        votes[1, 5] = 1  # Unknown owner vs known context.
        votes[2, 4] = 1
        votes[2, 7] = 1  # Known owner vs unknown context.
        votes[3, 7] = 2  # Unknown agrees with unknown.

        compared = compare_tile_proposals(owner, votes, ownership)
        self.assertEqual(compared["counts"]["known_class_disagreement_points"], 1)
        self.assertEqual(compared["counts"]["known_unknown_disagreement_points"], 2)
        self.assertEqual(compared["counts"]["cross_tile_disagreement_points"], 3)
        self.assertEqual(compared["counts"]["owner_unknown_without_known_context_points"], 1)
        self.assertIsNone(compared["counts"]["instance_id_only_differences"])
        self.assertFalse(compared["counts"]["instance_ids_compared_across_tiles"])
        np.testing.assert_array_equal(compared["disagreement_mask"], [True, True, True, False])

    def test_proposal_rejection_reason_bits_are_explicit(self):
        flags = proposal_rejection_flags(
            np.array([2, 7, 7, 7, 7, 7, 7], dtype=np.int8),
            np.array([4, 5, 5, 7, 5, 5, 7], dtype=np.int8),
            np.array([1, 2, 3, -1, 4, 4, 0], dtype=np.int32),
            np.array([1, 1, 1, 1, 0, 1, 1], dtype=np.uint8),
            np.array([True, True, True, True, True, False, True]),
            np.array([False, True, False, False, False, False, False]),
        )
        np.testing.assert_array_equal(flags, [1, 2, 0, 16, 8, 4, 32])

    def test_plane_column_and_line_cross_tiles_with_one_owner_each(self):
        gx, gy = np.meshgrid(np.linspace(0.0, 3.0, 16), np.linspace(0.0, 1.5, 8))
        plane = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
        plane_normals = np.tile([0.0, 0.0, 1.0], (len(plane), 1))

        theta, height = np.meshgrid(np.linspace(0, 2 * np.pi, 16, endpoint=False), np.linspace(0.1, 3.1, 16))
        column = np.column_stack((1.4 + 0.18 * np.cos(theta.ravel()),
                                  2.5 + 0.18 * np.sin(theta.ravel()), height.ravel()))
        column_normals = np.column_stack((np.cos(theta.ravel()), np.sin(theta.ravel()),
                                          np.zeros(theta.size)))

        line_x = np.linspace(0.0, 3.0, 40)
        line = np.column_stack((line_x, np.full_like(line_x, 4.0), np.full_like(line_x, 1.2)))
        line_normals = np.tile([0.0, 0.0, 1.0], (len(line), 1))
        points = np.vstack((plane, column, line))
        normals = np.vstack((plane_normals, column_normals, line_normals))
        truth = np.r_[np.zeros(len(plane), dtype=np.int8),
                      np.full(len(column), 3, dtype=np.int8),
                      np.full(len(line), 4, dtype=np.int8)]

        plan = build_spatial_tile_plan(points, tile_size=1.0, halo=0.2, max_tiles=100)
        preflight = preflight_tile_plan(
            plan, points, max_tile_points=500, max_edges=10_000,
            max_total_context_points=20_000,
        )
        self.assertGreater(plan.tile_count, 3)
        self.assertLessEqual(preflight["max_context_points"], 500)
        owner_count = np.zeros(len(points), dtype=np.uint8)
        owner_predictions = np.full(len(points), CLASS_IDS["unknown"], dtype=np.int8)
        owner_regions = np.full(len(points), -1, dtype=np.int32)
        votes = np.zeros((len(points), len(CLASS_IDS)), dtype=np.uint16)
        for tile_id in range(plan.tile_count):
            context = tile_context_indices(plan, points, tile_id)
            predicted = truth[context]
            np.add.at(votes, (context, predicted), 1)
            owned = plan.point_tile_ids[context] == tile_id
            rows = context[owned]
            owner_count[rows] += 1
            owner_predictions[rows] = predicted[owned]
            owner_regions[rows] = tile_id

        final, sources, metrics = finalize_tiled_predictions(
            np.full(len(points), CLASS_IDS["unknown"], dtype=np.int8),
            owner_predictions, owner_regions, owner_count, votes,
        )
        np.testing.assert_array_equal(owner_count, np.ones(len(points), dtype=np.uint8))
        np.testing.assert_array_equal(final, truth)
        self.assertEqual(metrics["cross_tile_disagreement_points"], 0)
        self.assertEqual(metrics["points_with_exactly_one_owner"], len(points))
        self.assertTrue(np.all(sources == SOURCE_IDS["local_region"]))
        self.assertGreater(preflight["total_context_points"], len(points))

    def test_cross_tile_conflict_becomes_unknown_but_known_v1_is_protected(self):
        v1 = np.array([7, 2, 7], dtype=np.int8)
        owner = np.array([5, 4, 7], dtype=np.int8)
        regions = np.array([0, 1, -1], dtype=np.int32)
        owners = np.ones(3, dtype=np.uint8)
        votes = np.zeros((3, len(CLASS_IDS)), dtype=np.uint16)
        votes[0, 5] = 1
        votes[0, 4] = 1  # The neighboring tile disagrees with the owner.
        votes[1, 4] = 2
        votes[2, 7] = 2
        final, source, metrics = finalize_tiled_predictions(v1, owner, regions, owners, votes)
        np.testing.assert_array_equal(final, [7, 2, 7])
        np.testing.assert_array_equal(source, [
            SOURCE_IDS["unknown"], SOURCE_IDS["v1"], SOURCE_IDS["unknown"],
        ])
        self.assertEqual(metrics["cross_tile_disagreement_points"], 1)
        self.assertEqual(metrics["known_v1_preserved"], 1)
        self.assertEqual(metrics["known_v1_predictions_overwritten"], 0)

    def test_tile_preflight_fails_before_processing_when_budget_is_exceeded(self):
        points = np.array([[x / 10, 0.0, 0.0] for x in range(20)])
        plan = build_spatial_tile_plan(points, tile_size=0.5, halo=0.1, max_tiles=100)
        with self.assertRaises(MemoryError):
            preflight_tile_plan(
                plan, points, max_tile_points=2, max_edges=100,
                max_total_context_points=100,
            )
        with self.assertRaises(MemoryError):
            build_spatial_tile_plan(points, tile_size=0.01, halo=0, max_tiles=2)

    def test_semantic_ply_cloud_keeps_xyz_and_normals(self):
        xyz = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
        normals = np.array([[0, 0, 1], [0, 1, 0]], dtype=float)
        source_rgb = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        before_xyz = xyz.copy()
        before_normals = normals.copy()
        cloud = make_semantic_cloud(xyz, normals, np.array([0, 7]))
        np.testing.assert_array_equal(np.asarray(cloud.points), before_xyz)
        np.testing.assert_array_equal(np.asarray(cloud.normals), before_normals)
        np.testing.assert_array_equal(np.asarray(cloud.colors), semantic_colors([0, 7]))
        np.testing.assert_array_equal(xyz, before_xyz)
        np.testing.assert_array_equal(normals, before_normals)
        np.testing.assert_array_equal(source_rgb, [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])

    def test_bad_roi_and_output_alignment_are_rejected(self):
        with self.assertRaises(ValueError):
            select_roi_indices(np.zeros((1, 3)), (np.ones(3), np.zeros(3)), max_points=5)
        with self.assertRaises(MemoryError):
            select_roi_indices(np.zeros((2, 3)), (np.full(3, -1), np.ones(3)), max_points=1)
        with self.assertRaises(ValueError):
            fuse_unknown_only(np.array([7]), np.array([0, 1]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
