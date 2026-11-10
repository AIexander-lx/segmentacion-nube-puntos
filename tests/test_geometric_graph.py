import unittest
import numpy as np
from segmentacion.features import local_pca_features
from segmentacion.geometric_graph import geometric_components

class GeometricGraphTests(unittest.TestCase):
 def test_two_planes_separated_by_normal_change(self):
  x=np.repeat(np.arange(6)*.04,6);y=np.tile(np.arange(6)*.04,6);a=np.c_[x,y,np.zeros_like(x)];b=np.c_[x+.24,y,np.zeros_like(x)];p=np.r_[a,b];n=np.r_[np.tile([0,0,1.],(len(a),1)),np.tile([1,0,0.],(len(b),1))]
  labels,_,_=geometric_components(p,n,eps=.09,max_neighbors=12,normal_angle_degrees=25,min_component_size=5)
  self.assertNotEqual(labels[0],labels[len(a)]);self.assertTrue(np.all(labels[:len(a)]>=0));self.assertTrue(np.all(labels[len(a):]>=0))
 def test_curved_surface_progressive_normals(self):
  t=np.linspace(0,.5,40);p=np.c_[t,np.zeros_like(t),np.zeros_like(t)];n=np.c_[np.sin(t),np.zeros_like(t),np.cos(t)]
  labels,_,_=geometric_components(p,n,eps=.03,max_neighbors=4,normal_angle_degrees=20,min_component_size=5)
  self.assertEqual(len(np.unique(labels[labels>=0])),1)
 def test_opposite_and_invalid_normals(self):
  p=np.array([[0,0,0],[.02,0,0],[.04,0,0],[.06,0,0]],float);n=np.array([[0,0,1],[0,0,-1],[0,0,0],[np.nan,0,0]],float)
  labels,metrics,_=geometric_components(p,n,eps=.05,max_neighbors=3,normal_angle_degrees=10,min_component_size=2)
  self.assertTrue(np.all(labels>=0));self.assertEqual(metrics["normal_rejected_edges"],0)
 def test_color_can_separate_nearby_objects_and_pca_optional(self):
  a=np.c_[np.arange(8)*.01,np.zeros(8),np.zeros(8)];b=np.c_[.12+np.arange(8)*.01,np.zeros(8),np.zeros(8)];p=np.r_[a,b];n=np.tile([0,0,1.],(len(p),1));c=np.r_[np.tile([1,0,0.],(8,1)),np.tile([0,0,1.],(8,1))]
  plain,_,_=geometric_components(p,n,eps=.06,max_neighbors=5,normal_angle_degrees=10,min_component_size=2)
  colored,_,_=geometric_components(p,n,eps=.06,max_neighbors=5,normal_angle_degrees=10,colors=c,color_distance=.2,min_component_size=2)
  self.assertEqual(plain[0],plain[8]);self.assertNotEqual(colored[0],colored[8])
  f=local_pca_features(p,k=5,batch_size=4);self.assertTrue(f.valid.all())
 def test_degenerate_pca_and_determinism_and_memory_cap(self):
  p=np.zeros((10,3));f=local_pca_features(p,k=5,batch_size=3);self.assertFalse(f.valid.any());self.assertTrue(np.isnan(f.values).all())
  p=np.c_[np.arange(20)*.01,np.zeros(20),np.zeros(20)];n=np.tile([0,0,1.],(20,1));a,_,_=geometric_components(p,n,eps=.04,max_neighbors=3,min_component_size=2);b,_,_=geometric_components(p,n,eps=.04,max_neighbors=3,min_component_size=2);np.testing.assert_array_equal(a,b)
  with self.assertRaises(MemoryError):geometric_components(p,n,eps=.1,max_neighbors=10,max_edges=20)

if __name__=="__main__":unittest.main()
