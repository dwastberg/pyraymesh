import cProfile
import pstats

import sys
from pathlib import Path

src_root = Path(__file__).resolve().parents[1] / "src"
print(f"src_root: {src_root}")
sys.path.append(str(src_root))

from pyraymesh import Mesh as prmMesh

try:
    from embreex import rtcore_scene as rtcs
    from embreex.mesh_construction import TriangleMesh

    EMBREE = True
except ImportError:
    EMBREE = False

import numpy as np
from pathlib import Path
import trimesh
from time import time
import timeit

bunny = Path(__file__).parent / "bunny.obj"
bunny = trimesh.load(bunny)
bunny_vertices = bunny.vertices
bunny_faces = bunny.faces


def gen_rays(bounds, num_rays):
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    np.random.seed(101)
    x = np.random.uniform(x_min, x_max, num_rays)
    y = np.random.uniform(y_min, y_max, num_rays)
    z = np.ones(num_rays) * (z_max + 10)
    return np.stack([x, y, z], axis=1).astype(np.float32)


rays = gen_rays(bunny.bounds, 1000)


def embree_bunny():
    scene = rtcs.EmbreeScene()
    mesh = TriangleMesh(scene, bunny_vertices, bunny_faces)
    return scene


def pyraymesh_bunny():
    mesh = prmMesh(bunny_vertices, bunny_faces)
    mesh.build()
    mesh.robust = True
    mesh.build("high")
    return mesh


def embree_intersect(scene, rays):
    res = scene.run(rays, np.array([[0, 0, -1]], dtype=np.float32))
    return res


def pyraymesh_intersect(mesh, rays):
    # mesh.robust = True
    # res = mesh.intersect(rays, np.array([0, 0, -1], dtype=np.float32))
    res = mesh.occlusion(rays, np.array([0, 0, -1], dtype=np.float32))
    return res


# embree_build_time = timeit.timeit(embree_bunny, number=10) / 10
# pyraymesh_build_time = timeit.timeit(pyraymesh_bunny, number=10) / 10

# print(f"Embree build time: {embree_build_time}")
# print(f"PyRayMesh build time: {pyraymesh_build_time}")
EMBREE = False
if EMBREE:
    embree_intersect_timer = timeit.Timer(
        lambda: embree_intersect(embree_bunny(), rays)
    )
    embree_intersect_time = embree_intersect_timer.timeit(number=10) / 10
    print(f"Embree intersect time: {embree_intersect_time}")
    res = embree_intersect(embree_bunny(), rays)
    print(f"embree len(res): {len(res)}")


# pyraymesh_intersect_timer = timeit.Timer(
#     lambda: pyraymesh_intersect(pyraymesh_bunny(), rays)
# )
# pyraymesh_intersect_time = pyraymesh_intersect_timer.timeit(number=1) / 1

start = time()
prm_mesh = prmMesh(bunny_vertices, bunny_faces)
prm_mesh.build()

# res = prm_mesh.intersect(rays, np.array([0, 0, -1], dtype=np.float32))
# print(f"num hits: {res.num_hits}")

res = prm_mesh.occlusion(np.array([[0, 8, 12]]), np.array([0, 0, -1], dtype=np.float32))
print(f"num hits: {res.sum()}")

print(f"PyRayMesh intersect time: {time() - start}")

# pyraymesh_intersect_timer = timeit.Timer(
#     lambda: pyraymesh_intersect(pyraymesh_bunny(), rays)
# )
# pyraymesh_intersect_time = pyraymesh_intersect_timer.timeit(number=1) / 1
# print(f"PyRayMesh intersect time: {pyraymesh_intersect_time}")
profiler = cProfile.Profile()
# profiler.enable()

start_time = time()
e_res = embree_intersect(embree_bunny(), rays)
print(f"embree time: {time() - start_time}")
print(f"embree_hits {len([res for res in e_res if res >= 0])}")

start_time = time()
res = pyraymesh_intersect(pyraymesh_bunny(), rays)
print(f"pyraymesh time: {time() - start_time}")
# profiler.disable()
print(f"pyraymesh hits: {res.sum()}")
# print(f"num hits: {res.num_hits}")
# stats = pstats.Stats(profiler).sort_stats("cumulative")
# stats.print_stats()
