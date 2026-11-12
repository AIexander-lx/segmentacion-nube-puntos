import unittest
import numpy as np
import open3d as o3d
from segmentacion.inspection import inspect_cloud
from segmentacion.diagnostics import geometric_diagnostics

def cloud(points, normals=None):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float).reshape((-1, 3)))
    if normals is not None:
        result.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float).reshape((-1, 3)))
    return result

class InspectionTests(unittest.TestCase):
    def test_regular_and_attributes(self):
        c=cloud([[0,0,0],[2,1,3]], [[0,0,1],[0,0,0]])
        c.colors=o3d.utility.Vector3dVector([[1,0,0],[0,1,0]])
        r=inspect_cloud(c)
        self.assertEqual(r["point_count"], 2); self.assertEqual(r["scene_dimensions"]["z"], 3); self.assertEqual(r["zero_normals"], 1)
    def test_empty(self):
        r=inspect_cloud(cloud([])); self.assertEqual(r["point_count"], 0); self.assertIsNone(r["largest_extent_axis_candidate"])
    def test_missing_attributes(self):
        r=inspect_cloud(cloud([[1,2,3]])); self.assertIsNone(r["colors_rgb"]); self.assertIsNone(r["normals"])
    def test_invalid_values(self):
        r=inspect_cloud(cloud([[np.nan,0,np.inf]])); self.assertEqual(r["invalid_values"]["xyz_nan_or_inf"], 2)
    def test_geometry_orientation_neighbors_and_voxels(self):
        points=np.array([[x,y,z] for x in np.linspace(0,1,5) for y in np.linspace(0,1,5) for z in (0.,0.1)])
        normals=np.tile([0.,0.,1.], (len(points),1))
        r=geometric_diagnostics(cloud(points,normals), sample_size=30, seed=7, voxels=(0.1,0.2))
        self.assertEqual(r["normal_orientation"]["horizontal_candidate_axis"], "z")
        self.assertEqual([v["voxel_size"] for v in r["voxel_comparison"]], [0.1,0.2])
        self.assertGreater(r["neighbor_sample"]["distance_quantiles"]["0.5"], 0)
        self.assertTrue(all(0 < v["point_count"] <= len(points) for v in r["voxel_comparison"]))

if __name__ == "__main__": unittest.main()
