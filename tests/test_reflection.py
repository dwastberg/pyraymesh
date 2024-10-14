import pytest

from pyraymesh import Mesh
import numpy as np

def mesh_plane():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)

def test_simple_reflecton():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [[0.5, 0.5, 1], [0.5, 0.5, -1]]
    ray_direction = [[0, 0, -1], [0, 0, 1]]
    result = m.intersect(ray_origin, ray_direction, calculate_reflections=True)
    assert len(result) == 2
    assert len(result.reflections) == 2
    assert np.allclose(result.reflections[0], [0, 0, 1])
    assert np.allclose(result.reflections[1], [0, 0, -1])

def test_angled_reflection():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [[0.5, 0.5, 1], [0.5, 0.5, -1]]
    ray_direction = [[-0.1,-0.1,-1], [-0.1,-0.1,1]]
    result = m.intersect(ray_origin, ray_direction, calculate_reflections=True)
    assert len(result) == 2
    assert len(result.reflections) == 2
    assert np.linalg.norm(result.reflections[0]) == 1
    assert np.linalg.norm(result.reflections[1]) == 1
    expected_reflection1 = np.array([-0.1, -0.1, 1])
    expected_reflection2 = np.array([-0.1, -0.1, -1])
    expected_reflection1 /= np.linalg.norm(expected_reflection1)
    expected_reflection2 /= np.linalg.norm(expected_reflection2)

    assert np.allclose(result.reflections[0], expected_reflection1)
    assert np.allclose(result.reflections[1], expected_reflection2)

def test_miss():
    m = mesh_plane()
    m.build()
    ray_origin = [[0.5, 0.5, 1]]
    ray_direction = [[0, 0, 1]]
    result = m.intersect(ray_origin, ray_direction, calculate_reflections=True)
    assert result.num_hits == 0
    assert len(result.reflections) == 1
    assert np.all(np.isnan(result.reflections[0]))