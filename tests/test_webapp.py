import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import open3d as o3d

from segmentacion.webapp import Project, ROOT, create_server
from segmentacion.io import load_ply, save_labeled_ply


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        for directory in ("outputs/processed", "outputs/segmented/global_review_final", "outputs/semantic/global_heuristic_v1", "outputs/semantic/v3_test_release", "outputs/reconstruction/test", "outputs/app_runs/roi"):
            (root / directory).mkdir(parents=True)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float))
        cloud.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (4, 1)))
        cloud.colors = o3d.utility.Vector3dVector(np.tile([1., 0., 0.], (4, 1)))
        for path in (root / "FARO_Scan_Full_Sub1_down3_with_normals.ply", root / "outputs/processed/global.ply", root / "outputs/processed/detail.ply"):
            assert o3d.io.write_point_cloud(str(path), cloud)
        np.savez(root / "outputs/processed/detail_mapping.npz", voxel_to_representative=np.array([0, 1, 2, 3]))
        np.savez(root / "outputs/segmented/global_review_final/labels.npz", original_labels=np.array([0, -1, 1, -1]), global_labels=np.array([0, -1, 1, -1]))
        np.savez(root / "outputs/semantic/global_heuristic_v1/semantic_labels.npz", semantic_labels_original=np.array([0, 7, 5, 7]), semantic_labels_global=np.array([0, 7, 5, 7]))
        np.savez(root / "outputs/semantic/v3_test_release/semantic_v3_labels.npz",
                 semantic_original=np.array([0, 7, 5, 7]), semantic_global=np.array([0, 7, 5, 7]),
                 instance_original=np.array([0, -1, 1, -1]), instance_global=np.array([0, -1, 1, -1]))
        (root / "outputs/semantic/v3_test_release/semantic_v3_original.ply").write_bytes((root / "FARO_Scan_Full_Sub1_down3_with_normals.ply").read_bytes())
        (root / "outputs/semantic/v3_test_release/semantic_v3_metrics.json").write_text(
            json.dumps({"comparison_v3": {"original_coverage_percent": 50.0}, "class_counts_original": {"floor": 1}}), encoding="utf-8")
        np.savez(root / "outputs/app_runs/roi/hybrid_tiles_labels.npz", global_indices=np.array([1]), original_indices=np.array([1]), semantic_labels_v1_global=np.array([7]), semantic_labels_v2_global=np.array([4]), semantic_labels_v1_original=np.array([7]), semantic_labels_v2_original=np.array([4]))
        (root / "outputs/app_runs/roi/hybrid_tiles_metrics.json").write_text("{}", encoding="utf-8")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float))
        mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], int))
        assert o3d.io.write_triangle_mesh(str(root / "outputs/reconstruction/test/ball_pivoting.ply"), mesh)
        cls.project = Project(root)
        cls.server = create_server(cls.project)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.temp.cleanup()

    def test_metadata_and_binary_clouds(self):
        with urlopen(self.url + "/") as response:
            self.assertIn("SEGMENTACIÓN DE NUBE DE PUNTOS 3D", response.read().decode("utf-8"))
        with urlopen(self.url + "/app.js") as response:
            self.assertIn("webgl", response.read().decode("utf-8"))
        with urlopen(self.url + "/api/metadata") as response:
            metadata = json.load(response)
        self.assertTrue(all(metadata["clouds"].values()))
        self.assertTrue(metadata["semantic_available"])
        with urlopen(self.url + "/api/cloud?resolution=detail&mode=semantic&limit=4&hide_unknown=1") as response:
            points = np.frombuffer(response.read(), dtype="<f4").reshape(-1, 6)
            self.assertEqual(response.headers["X-Source-Count"], "4")
        self.assertEqual(len(points), 2)
        self.assertEqual({tuple(x) for x in points[:, :3]}, {(0., 0., 0.), (0., 1., 0.)})
        with urlopen(self.url + "/api/cloud?resolution=original&mode=semantic&limit=4&result=v3_test_release") as response:
            v3 = np.frombuffer(response.read(), dtype="<f4").reshape(-1, 6)
        self.assertEqual(len(v3), 4)
        with urlopen(self.url + "/api/export?result=v3_test_release") as response:
            self.assertIn("semantic_v3_original.ply", response.headers["Content-Disposition"])

    def test_project_finds_original_in_data_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "data/raw/FARO_Scan_Full_Sub1_down3_with_normals.ply"
            raw_path.parent.mkdir(parents=True)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
            self.assertTrue(o3d.io.write_point_cloud(str(raw_path), cloud))
            project = Project(root)
            self.assertEqual(project.path(ROOT / "FARO_Scan_Full_Sub1_down3_with_normals.ply"), raw_path)
            self.assertTrue(project.metadata()["clouds"]["original"])

    def test_mesh_and_missing_paths(self):
        with urlopen(self.url + "/api/mesh?name=reconstruction%2Ftest%2Fball_pivoting.ply") as response:
            triangles = np.frombuffer(response.read(), dtype="<f4")
        self.assertEqual(len(triangles), 9)
        with self.assertRaises(HTTPError) as missing:
            urlopen(self.url + "/api/mesh?name=..%2Fsecret")
        self.assertEqual(missing.exception.code, 404)
        with self.assertRaises(HTTPError) as bad:
            urlopen(self.url + "/api/cloud?resolution=original&limit=500001")
        self.assertEqual(bad.exception.code, 400)

    def test_saved_roi_result_is_aligned_with_baseline(self):
        body, info = self.project.cloud("global", "semantic", 4, False, "roi")
        points = np.frombuffer(body, dtype="<f4").reshape(-1, 6)
        by_xyz = {tuple(row[:3]): row[3:] for row in points}
        np.testing.assert_allclose(by_xyz[(1., 0., 0.)], [0.75, 0.25, 0.80])
        self.assertEqual(info["shown"], 4)
        self.assertIn("roi", self.project.results())

    def test_external_ply_requires_its_own_labels_and_restores_project(self):
        source = self.project.root / "FARO_Scan_Full_Sub1_down3_with_normals.ply"
        try:
            self.project.set_input(str(source))
            self.assertFalse(self.project.metadata()["clouds"]["global"])
            self.assertEqual(self.project.cloud("original", "rgb", 4, False)[1]["shown"], 4)
            with self.assertRaises(ValueError): self.project.cloud("original", "semantic", 4, False)
            with self.assertRaises(ValueError): self.project.start_job({"kind": "sample_segmentation"})
        finally:
            self.project.set_input("")
        self.assertTrue(self.project.metadata()["clouds"]["global"])

    def test_jobs_require_token_and_reject_invalid_limits(self):
        request = Request(self.url + "/api/jobs", data=b'{"kind":"sample_segmentation","sample":9000}', method="POST", headers={"X-Project-Token": self.project.token})
        with self.assertRaises(HTTPError) as bad: urlopen(request)
        self.assertEqual(bad.exception.code, 400)
        request = Request(self.url + "/api/jobs", data=b'{}', method="POST")
        with self.assertRaises(HTTPError) as forbidden: urlopen(request)
        self.assertEqual(forbidden.exception.code, 403)

    def test_v3_job_is_explicit_and_writes_to_a_new_semantic_directory(self):
        from unittest.mock import patch
        with patch("segmentacion.webapp.threading.Thread") as thread:
            job = self.project.start_job({"kind": "semantic_v3"})
            command = thread.call_args.kwargs["args"][0]
            self.assertEqual(job["state"], "running")
            self.assertIn("outputs/semantic/v3_", job["output"].replace("\\", "/"))
            self.assertTrue(any(argument.replace("\\", "/").endswith("scripts/segment_v3.py") for argument in command))
            self.assertIn("--max-edges", command)
            self.assertIn("--max-memory-mb", command)
            thread.return_value.start.assert_called_once()
        self.project.job = {"state": "idle", "message": "Sin procesamiento activo"}

    def test_colored_export_preserves_source_coordinates_and_normals(self):
        source_path = self.project.root / "FARO_Scan_Full_Sub1_down3_with_normals.ply"
        source = load_ply(source_path)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "colored.ply"
            save_labeled_ply(source, np.array([0, 1, -1, 1]), np.array([[1., 0., 0.], [0., 1., 0.]]), target)
            exported = load_ply(target)
            np.testing.assert_array_equal(np.asarray(source.points), np.asarray(exported.points))
            np.testing.assert_array_equal(np.asarray(source.normals), np.asarray(exported.normals))
            self.assertEqual(len(exported.points), 4)

    def test_launcher_help(self):
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--no-browser", result.stdout)


if __name__ == "__main__": unittest.main()
