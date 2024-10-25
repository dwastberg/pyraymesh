import pytest

from pyraymesh import Mesh
import numpy as np


def mesh_plane():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)

def test_fars():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [[0.1, 0.2, 1], [0.3, 0.4, 1],[0.5, 0.6, 1]]
    ray_direction = [0, 0, -1]
    tnear = 0
    tfar = [0.5, 0.6, 1.5]
    result = m.intersect(ray_origin, ray_direction, tnear, tfar)
    assert len(result.coords) == 3
    assert result.num_hits == 1
    assert np.all(result.hit_mask == [False, False, True])

def test_nears():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [[0.1, 0.2, 1], [0.3, 0.4, 1],[0.5, 0.6, 1]]
    ray_direction = [0, 0, -1]
    tnear = [0.1, 0.2, 1.3]
    result = m.intersect(ray_origin, ray_direction, tnear)
    assert len(result.coords) == 3
    assert result.num_hits == 2
    assert np.all(result.hit_mask == [True, True, False])

