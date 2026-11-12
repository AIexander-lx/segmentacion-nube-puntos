import tempfile,unittest
from pathlib import Path
import numpy as np,open3d as o3d
from segmentacion.reconstruction import prepare_oriented_cloud,estimate_spacing,reconstruct_bpa,reconstruct_poisson,clean_mesh,mesh_quality,vertex_source_mapping,filter_supported_mesh

def make_cloud(points,normals,colors=None):
 c=o3d.geometry.PointCloud();c.points=o3d.utility.Vector3dVector(np.asarray(points,float));c.normals=o3d.utility.Vector3dVector(np.asarray(normals,float))
 if colors is not None:c.colors=o3d.utility.Vector3dVector(np.asarray(colors,float))
 return c

class ReconstructionTests(unittest.TestCase):
 def test_support_filter_rejects_unobserved_long_triangle(self):
  points=np.array([[0,0,0],[1,0,0],[0,1,0]],float);normals=np.tile([0,0,1.],(3,1));mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector(np.vstack((points,[[8,8,0]])));mesh.triangles=o3d.utility.Vector3iVector(np.array([[0,1,2],[0,1,3]]));filtered,stats=filter_supported_mesh(mesh,points,normals,max_edge=2,max_distance=.1);self.assertEqual(len(filtered.triangles),1);self.assertEqual(stats['removed'],1);self.assertIn('inferida',stats['interpretation'])
 def test_planar_crop_and_bpa_mesh_export(self):
  x,y=np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,12));p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)];n=np.tile([0,0,1.],(len(p),1));cloud=make_cloud(p,n,np.tile([.2,.4,.7],(len(p),1)))
  prep=prepare_oriented_cloud(cloud,orient=True,neighbor_k=10);self.assertEqual(len(prep.source_indices),len(p));self.assertTrue(prep.cloud.has_colors());self.assertGreater(estimate_spacing(prep.cloud),0)
  mesh,_=reconstruct_bpa(prep.cloud,[.2,.35]);self.assertGreater(len(mesh.triangles),0)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"mesh.ply";self.assertTrue(o3d.io.write_triangle_mesh(str(path),mesh));loaded=o3d.io.read_triangle_mesh(str(path));self.assertEqual(len(loaded.triangles),len(mesh.triangles))
 def test_curved_poisson_and_quality(self):
  rng=np.random.default_rng(2);v=rng.normal(size=(300,3));v/=np.linalg.norm(v,axis=1,keepdims=True);cloud=make_cloud(v,v);prep=prepare_oriented_cloud(cloud,neighbor_k=20)
  mesh,stats,_=reconstruct_poisson(prep.cloud,depth=5,density_quantile=.05);self.assertGreater(len(mesh.triangles),0);self.assertGreater(stats["vertices_removed_low_density"],0);self.assertIsNotNone(mesh_quality(mesh,v)["distance_p95"])
  indices,dist=vertex_source_mapping(mesh,v);self.assertEqual(len(indices),len(mesh.vertices));self.assertTrue(np.isfinite(dist).all())
 def test_invalid_normals_excluded_without_invention(self):
  cloud=make_cloud([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[.5,.5,0],[.2,.3,0],[.4,.2,0],[.8,.4,0],[.7,.8,0],[.9,.1,0],[.3,.8,0],[.6,.6,0]],[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,0],[np.nan,0,0],[0,0,1],[0,0,1]])
  prep=prepare_oriented_cloud(cloud,orient=False);self.assertEqual(prep.excluded_invalid_normals,2);self.assertEqual(len(prep.source_indices),10);self.assertEqual(prep.source_indices[-1],11)
 def test_empty_and_bad_parameters(self):
  empty=o3d.geometry.PointCloud();mesh=clean_mesh(o3d.geometry.TriangleMesh());self.assertEqual(len(mesh.triangles),0)
  with self.assertRaises(ValueError):reconstruct_bpa(empty,[.1])
  with self.assertRaises(ValueError):reconstruct_poisson(empty,depth=20)
  with self.assertRaises(ValueError):prepare_oriented_cloud(empty)
  with self.assertRaises(ValueError):estimate_spacing(empty)

if __name__=="__main__":unittest.main()
