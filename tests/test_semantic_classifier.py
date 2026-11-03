import unittest
import numpy as np
from segmentacion.semantic_features import region_features
from segmentacion.semantic_classifier import CLASS_IDS,RuleConfig,classify_region,propagate_semantic_labels,semantic_colors

SCENE=(np.array([-3.,-3.,-3.]),np.array([3.,3.,3.]))
def grid_plane(axis,position,size=12):
 q=np.linspace(-1,1,size);a,b=np.meshgrid(q,q);p=np.zeros((a.size,3));others=[i for i in range(3) if i!=axis];p[:,others[0]]=a.ravel();p[:,others[1]]=b.ravel();p[:,axis]=position;n=np.zeros_like(p);n[:,axis]=1;return p,n
def region(points,normals,kind="cluster",plane=None,colors=None):
 f=region_features(points,normals=normals,colors=colors,scene_bounds=SCENE,vertical_axis=2);return classify_region(f,region_type=kind,plane_model=plane,vertical_axis_confirmed=True)

class SemanticTests(unittest.TestCase):
 def test_floor_ceiling_wall(self):
  for axis,z,label,pid in [(2,-2.8,"floor",0),(2,2.8,"ceiling",1),(0,-2.8,"wall",2)]:
   p,n=grid_plane(axis,z);model={"id":pid,"normal":n[0].tolist(),"equation":[*n[0],-z]};r=region(p,n,"plane",model);self.assertEqual(r["class"],label)
 def test_column_beam_and_pipe(self):
  z=np.linspace(0,3,30);x=np.linspace(-.2,.2,5);y=np.linspace(-.1,.1,5);zz,xx,yy=np.meshgrid(z,x,y,indexing="ij");column=np.c_[xx.ravel(),yy.ravel(),zz.ravel()];cn=np.tile([1.,0,0],(len(column),1));self.assertEqual(region(column,cn)["class"],"column")
  x=np.linspace(-2,2,300);beam=np.c_[x,np.zeros_like(x),np.zeros_like(x)];bn=np.tile([0,0,1.],(len(x),1));self.assertEqual(region(beam,bn)["class"],"beam")
  x,theta=np.meshgrid(np.linspace(-1,1,30),np.linspace(0,2*np.pi,24,endpoint=False));pipe=np.c_[x.ravel(),.2*np.cos(theta.ravel()),.2*np.sin(theta.ravel())];pn=np.c_[np.zeros(theta.size),np.cos(theta.ravel()),np.sin(theta.ravel())];self.assertEqual(region(pipe,pn)["class"],"pipe")
  strict=classify_region(region_features(pipe,normals=pn,scene_bounds=SCENE),config=RuleConfig(require_tube_radial_evidence=True));self.assertEqual(strict["class"],"pipe")
 def test_ambiguous_mixed_and_missing(self):
  x=np.linspace(-1,1,30);p=np.r_[np.c_[x,np.zeros_like(x),np.zeros_like(x)],np.c_[np.zeros_like(x),x,np.zeros_like(x)]];n=np.tile([0,0,1.],(len(p),1));self.assertEqual(region(p,n)["class"],"unknown")
  f=region_features(np.empty((0,3)));self.assertEqual(classify_region(f)["class"],"unknown")
  f=region_features(np.zeros((10,3)));self.assertEqual(classify_region(f)["class"],"unknown")
 def test_scores_are_heuristics_and_use_rgb_only_auxiliary(self):
  p,n=grid_plane(2,-2.8);colors=np.tile([.1,.2,.3],(len(p),1));f=region_features(p,normals=n,colors=colors,scene_bounds=SCENE);r=classify_region(f,region_type="plane",plane_model={"normal":[0,0,1]},vertical_axis_confirmed=False)
  self.assertFalse(r["scores_are_probabilities"]);self.assertIn("rgb_auxiliary",f);self.assertIn("vertical_axis_is_hypothesis",r["warnings"])
 def test_propagation_colors_and_bad_mapping(self):
  labels=np.array([0,5,7]);mapping=np.array([0,1,1,2,-1]);out=propagate_semantic_labels(labels,mapping);np.testing.assert_array_equal(out,[0,5,5,7,7]);self.assertEqual(semantic_colors(out).shape,(5,3));np.testing.assert_array_equal(semantic_colors(out),semantic_colors(out))
  with self.assertRaises(ValueError):propagate_semantic_labels(labels,np.array([4]))
 def test_invalid_rule_thresholds(self):
  with self.assertRaises(ValueError):RuleConfig(decision_min_score=1.1)
  with self.assertRaises(ValueError):RuleConfig(cross_section_ratio_max=0)
  with self.assertRaises(ValueError):RuleConfig(pipe_angular_coverage_min=1.1)

if __name__=="__main__":unittest.main()
