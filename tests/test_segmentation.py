import unittest
import numpy as np
from segmentacion.planes import extract_planes
from segmentacion.clustering import spatial_components
from segmentacion.preprocessing import propagate_voxel_labels

class SegmentationTests(unittest.TestCase):
 def test_plane_with_noise(self):
  rng=np.random.default_rng(5); xy=rng.uniform(-1,1,(300,2)); plane=np.c_[xy,np.zeros(300)]; noise=rng.uniform(-1,1,(40,3)); points=np.r_[plane,noise]
  labels,models,_=extract_planes(points,.01,300,200,2,seed=12)
  self.assertEqual(len(models),1);self.assertGreaterEqual(models[0]["point_count"],280);self.assertGreater(abs(models[0]["normal"][2]),.99)
 def test_two_spatial_groups_and_noise(self):
  rng=np.random.default_rng(3); a=rng.normal([0,0,0],.02,(60,3));b=rng.normal([2,0,0],.02,(60,3));pts=np.r_[a,b,[[10,10,10]]]
  labels,stats,_=spatial_components(pts,.12,3,12,20,100)
  self.assertEqual(stats["clusters"],2);self.assertNotEqual(labels[0],labels[60]);self.assertEqual(labels[-1],-1)
 def test_ids_and_mapping(self):
  plane=np.c_[np.random.default_rng(2).random((100,2)),np.zeros(100)];groups=np.r_[np.random.default_rng(1).normal([3,0,0],.01,(40,3)),np.random.default_rng(4).normal([4,0,0],.01,(40,3))]
  pl,models,_=extract_planes(plane,.01,200,80,1,7);rest=groups;cl,stats,_=spatial_components(rest,.1,3,10,10,200)
  plane_id=0;cluster_base=len(models);self.assertGreater(cluster_base+int(cl.max()),plane_id)
  mapping=np.array([0,0,1,1,-1]);labels=propagate_voxel_labels(np.array([8,9]),mapping);np.testing.assert_array_equal(labels,[8,8,9,9,-1])
 def test_border_assignment_and_determinism(self):
  core=np.array([[x,y,0.] for x in (0,.04,.08,.12,.16) for y in (0,.04,.08,.12,.16)])
  pts=np.r_[core,[[-.05,.08,0],[.6,.6,0]]]
  first,metrics,_=spatial_components(pts,.061,4,16,3,100)
  again,_,_=spatial_components(pts,.061,4,16,3,100)
  np.testing.assert_array_equal(first,again)
  self.assertEqual(first[25],first[0]); self.assertEqual(first[26],-1)
  self.assertEqual(metrics["border_points"],1); self.assertEqual(metrics["noise_points"],1)
 def test_border_neighbor_limit(self):
  pts=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.02,.01,0],[.10,0,0]])
  labels,metrics,_=spatial_components(pts,.11,3,4,3,20)
  self.assertGreaterEqual(labels[-1],0); self.assertLessEqual(metrics["oversized_neighbor_rows"],len(pts))
 def test_empty_and_invalid(self):
  labels,models,_=extract_planes(np.empty((0,3)),max_planes=0);self.assertEqual(len(models),0)
  with self.assertRaises(ValueError):extract_planes(np.zeros((5,3)),distance_threshold=0)
  with self.assertRaises(ValueError):spatial_components(np.zeros((3,3)),max_neighbors=2,min_points=3)

if __name__=="__main__":unittest.main()
