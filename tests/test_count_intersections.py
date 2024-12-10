import pytest

from pyraymesh import Mesh
import numpy as np

def mesh_N_planes(n = 1):
    vertice = []
    faces = []
    for z in range(n):
        vertice += [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]
        faces += [[0+(z*4), 1+(z*4), 2+(z*4)], [2+(z*4), 3+(z*4), 0+(z*4)]]
    vertices = np.array(vertice)
    faces = np.array(faces)
    return Mesh(vertices, faces)

def test_one_intersection():
    ray_origin = [[0.6, 0.7, 100],[0.7,0.6,100]]
    ray_direction = [0, 0, -1]
    mesh = mesh_N_planes(1)
    result = mesh.count_intersections(ray_origin, ray_direction)
    assert len(result) == 2
    assert result[0] == 1
    assert result[1] == 1

def test_no_intersection():
    ray_origin = [[0.6, 0.7, 100],[0.7,0.6,100]]
    ray_direction = [0, 0, 1]
    mesh = mesh_N_planes(1)
    result = mesh.count_intersections(ray_origin, ray_direction)
    assert len(result) == 2
    assert result[0] == 0
    assert result[1] == 0

def test_many_intersections():
    ray_origin = [[0.6, 0.7, 1000],[0.7,0.6,1000]]
    ray_direction = [0, 0, -1]
    mesh = mesh_N_planes(500)
    result = mesh.count_intersections(ray_origin, ray_direction)
    assert len(result) == 2
    assert result[0] == 500
    assert result[1] == 500

