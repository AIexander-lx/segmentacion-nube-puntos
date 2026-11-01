import unittest
import numpy as np
import open3d as o3d
from segmentacion.preprocessing import validate_xyz, voxel_reduce, propagate_voxel_labels

def cloud(points, colors=None, normals=None):
    c=o3d.geometry.PointCloud(); c.points=o3d.utility.Vector3dVector(np.asarray(points,dtype=float).reshape((-1,3)))
    if colors is not None: c.colors=o3d.utility.Vector3dVector(np.asarray(colors,dtype=float))
    if normals is not None: c.normals=o3d.utility.Vector3dVector(np.asarray(normals,dtype=float))
    return c

class PreprocessingTests(unittest.TestCase):
    def test_reduction_attributes_ranges_and_original_unchanged(self):
        pts=[[0,0,0],[.01,0,0],[1,1,1]]; c=cloud(pts,[[1,0,0],[0,1,0],[0,0,1]],[[0,0,2],[0,0,4],[0,0,0]])
        before=np.asarray(c.points).copy(); r=voxel_reduce(c,.1)
        self.assertEqual(len(r.cloud.points),2); np.testing.assert_array_equal(np.asarray(c.points),before)
        self.assertTrue(r.cloud.has_colors()); self.assertTrue(r.cloud.has_normals())
        self.assertAlmostEqual(np.linalg.norm(np.asarray(r.cloud.normals)[0]),1); self.assertFalse(r.normal_valid[1])
        self.assertGreaterEqual(np.asarray(r.cloud.points).min(),0); self.assertLessEqual(np.asarray(r.cloud.points).max(),1)
    def test_invalid_xyz_mapping_and_normals(self):
        c=cloud([[0,0,0],[np.nan,1,1],[.01,0,0]],normals=[[0,0,0],[0,0,1],[np.nan,0,0]])
        self.assertEqual(validate_xyz(c).tolist(),[True,False,True]); r=voxel_reduce(c,.1)
        self.assertEqual(r.original_to_voxel.tolist(),[0,-1,0]); self.assertFalse(r.normal_valid[0])
        self.assertEqual(r.voxel_to_representative.tolist(),[0])
    def test_determinism_and_label_propagation(self):
        c=cloud([[0,0,0],[.01,0,0],[1,0,0]])
        a=voxel_reduce(c,.1); b=voxel_reduce(c,.1)
        np.testing.assert_array_equal(a.original_to_voxel,b.original_to_voxel); np.testing.assert_array_equal(a.voxel_to_representative,b.voxel_to_representative)
        np.testing.assert_array_equal(propagate_voxel_labels(np.array([7,3]),a.original_to_voxel),[7,7,3])
    def test_bad_voxel_and_mapping(self):
        c=cloud([[0,0,0]])
        for size in (0,-1,np.inf):
            with self.assertRaises(ValueError): voxel_reduce(c,size)
        with self.assertRaises(ValueError): propagate_voxel_labels(np.array([1]),np.array([2]))

if __name__=="__main__": unittest.main()
