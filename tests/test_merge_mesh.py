from pyraymesh import Mesh

import numpy as np


def mesh_plane(z=0):
    vertices = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)


def test_merge_empty():
    base_mesh = mesh_plane(0)
    mesh1 = mesh_plane(0)
    mesh2 = Mesh(vertices=[], faces=[])
    mesh1.merge(mesh2)
    assert len(mesh1.vertices) == 4
    assert len(mesh1.faces) == 2
    assert np.all(mesh1.vertices == base_mesh.vertices)
    assert np.all(mesh1.faces == base_mesh.faces)


def test_merge_meshes():
    mesh1 = mesh_plane(0)
    mesh2 = mesh_plane(1)
    mesh1.merge(mesh2)
    assert len(mesh1.vertices) == 8
    assert len(mesh1.faces) == 4
    assert np.all(mesh1.faces[2] == [4, 5, 6])
    assert np.all(mesh1.faces[3] == [6, 7, 4])
