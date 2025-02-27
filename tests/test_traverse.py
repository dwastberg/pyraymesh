import pytest
from numpy.f2py.symbolic import as_ref

from pyraymesh import Mesh
import numpy as np


def mesh_planes():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 10],
            [1, 0, 10],
            [1, 1, 10],
            [0, 1, 10],
        ]
    )
    faces = np.array([[0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4]])
    return Mesh(vertices, faces)


def test_traverse_all():
    m = mesh_planes()
    m.build("medium")
    ray_origin = [0.5, 0.5, 0.5]
    ray_direction = [0, 0, -1]

    travesal = m.traverse_all(ray_origin, ray_direction)
    assert len(travesal) == 2
    assert 0 in travesal
    assert 1 in travesal
    assert 2 not in travesal
    assert 3 not in travesal


def test_traverse():
    m = mesh_planes()
    m.build("medium")
    ray_origin = [0.5, 0.5, 0.5]
    ray_direction = [0, 0, -1]

    travesed = []
    for t_id in m.traverse(ray_origin, ray_direction):
        travesed.append(t_id)
    assert len(travesed) == 2
    assert 0 in travesed
    assert 1 in travesed
    assert 2 not in travesed
    assert 3 not in travesed


def test_traverse_all_r():
    m = mesh_planes()
    m.build("medium")
    ray_origin = [0.5, 0.5, 0.5]
    ray_direction = [0, 0, 1]

    travesal = m.traverse_all(ray_origin, ray_direction)
    assert len(travesal) == 2
    assert 0 not in travesal
    assert 1 not in travesal
    assert 2 in travesal
    assert 3 in travesal


def test_traverse_r():
    m = mesh_planes()
    m.build("medium")
    ray_origin = [0.5, 0.5, 0.5]
    ray_direction = [0, 0, 1]

    travesed = []
    for t_id in m.traverse(ray_origin, ray_direction):
        travesed.append(t_id)
    assert len(travesed) == 2
    assert 0 not in travesed
    assert 1 not in travesed
    assert 2 in travesed
    assert 3 in travesed
