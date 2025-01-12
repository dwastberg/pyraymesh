import pytest
import numpy as np

from pyraymesh import Mesh
from time import time


def mesh_plane():
    vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)


def build_ray(num_rays):
    ray_origin = np.random.rand(num_rays, 3) * 12
    ray_direction = [0, 0, -1]
    return ray_origin, ray_direction


def test_multithreaded_intersect():
    m = mesh_plane()
    m.build("medium")
    num_rays = 100
    ray_origin, ray_direction = build_ray(num_rays)
    result_1t = m.intersect(ray_origin, ray_direction)
    result_mt = m.intersect(ray_origin, ray_direction, threads=-1)

    assert len(result_1t) == num_rays
    assert len(result_mt) == num_rays

    assert result_1t.num_hits == result_mt.num_hits
    assert np.allclose(result_1t.distances, result_mt.distances, equal_nan=True)
    assert np.allclose(result_1t.coords, result_mt.coords, equal_nan=True)


def test_multithreaded_occlusion():
    m = mesh_plane()
    m.build("medium")
    num_rays = 100
    ray_origin, ray_direction = build_ray(num_rays)
    result_1t = m.occlusion(ray_origin, ray_direction)
    result_mt = m.occlusion(ray_origin, ray_direction, threads=-1)

    assert len(result_1t) == num_rays
    assert len(result_mt) == num_rays

    assert np.allclose(result_1t, result_mt)


def _test_perf_multithreaded_intersect():
    m = mesh_plane()
    m.build("medium")
    num_rays = 10000
    ray_origin, ray_direction = build_ray(num_rays)
    t_time = time()
    result_1t = m.intersect(ray_origin, ray_direction, threads=1)
    time_1t = time() - t_time

    t_time = time()
    result_mt = m.intersect(ray_origin, ray_direction, threads=4)
    time_mt = time() - t_time

    time_ratio = time_1t / time_mt
    assert time_ratio > 1.5


def _test_perf_multithreaded_occlusion():
    m = mesh_plane()
    m.build("medium")
    num_rays = 10000
    ray_origin, ray_direction = build_ray(num_rays)
    t_time = time()
    result_1t = m.occlusion(ray_origin, ray_direction, threads=1)
    time_1t = time() - t_time

    t_time = time()
    result_mt = m.occlusion(ray_origin, ray_direction, threads=4)
    time_mt = time() - t_time

    time_ratio = time_1t / time_mt
    assert time_ratio > 1.5
