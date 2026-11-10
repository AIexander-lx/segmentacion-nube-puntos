import tempfile, unittest
from pathlib import Path
import numpy as np
from segmentacion.visualization import deterministic_palette,label_colors,normals_colors,roi_indices,save_roi,labels_for_resolution
from segmentacion.evaluation import create_annotation,read_annotation,evaluate_instances,compare_runs

class VisualEvaluationTests(unittest.TestCase):
 def test_roi_indices_and_original_mapping(self):
  points=np.array([[0,0,0],[1,1,1],[2,2,2]],float);ids=roi_indices(points,(0,1,0,1,0,1));np.testing.assert_array_equal(ids,[0,1])
  with tempfile.TemporaryDirectory() as d:
   out=save_roi(points,ids,Path(d)/"roi.npz",resolution="global",representative_original=np.array([5,6,7]),original_to_voxel=np.array([0,0,1,2]))
   with np.load(out) as data:
    np.testing.assert_array_equal(data["representative_original_indices"],[5,6]);np.testing.assert_array_equal(data["original_indices"],[0,1,2]);np.testing.assert_array_equal(data["original_to_voxel"],[0,0,1])
 def test_colors_and_label_alignment(self):
  np.testing.assert_array_equal(deterministic_palette(3),deterministic_palette(3));colors=label_colors(np.array([0,-1,1]));np.testing.assert_array_equal(colors[1],[.45,.45,.45]);self.assertEqual(normals_colors(np.array([[0,0,1],[0,0,0]])).shape,(2,3))
  with tempfile.TemporaryDirectory() as d:
   lp=Path(d)/"labels.npz";mp=Path(d)/"map.npz";np.savez(lp,global_labels=np.array([4,5]),original_labels=np.array([4,4,5]));np.savez(mp,original_to_voxel=np.array([0,0,1]),voxel_to_representative=np.array([1,2]))
   np.testing.assert_array_equal(labels_for_resolution(lp,"detail",3,2,mp),[4,5])
 def test_annotation_and_instance_iou(self):
  with tempfile.TemporaryDirectory() as d:
   path=create_annotation(Path(d)/"ann.json",np.arange(4),region="test",categories=["part"]);ann=read_annotation(path)
   for i,row in enumerate(ann["points"]):row["instance_id"]="A" if i<2 else "B"
   result=evaluate_instances(ann,np.array([3,3,8,-1]));self.assertAlmostEqual(result["mean_iou"],.75);self.assertEqual(result["fragmentation_by_manual_instance"]["A"],1)
   self.assertIsNone(read_annotation(path)["points"][0]["category"])
 def test_compare_does_not_assume_equal_ids(self):
  result=compare_runs(np.array([0,-1,0]),np.array([8,-1,8]));self.assertFalse(result["ids_compared_as_equivalent"]);self.assertEqual(result["both_assigned"],2)
  with self.assertRaises(ValueError):compare_runs(np.array([0]),np.array([0,1]))
 def test_invalid_roi(self):
  with self.assertRaises(ValueError):roi_indices(np.zeros((1,3)),(2,1,0,1,0,1))

if __name__=="__main__":unittest.main()
