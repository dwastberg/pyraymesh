import pytest

from pyraymesh import Mesh
import numpy as np


def mesh_plane():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)


def test_import_Mesh():
    m = Mesh([], [])
    assert m is not None
    assert isinstance(m, Mesh)

def test_empty_mesh():
    m = Mesh([], [])
    with pytest.raises(ValueError):
        m.build("medium")

def test_build_low_bvh():
    m = mesh_plane()
    m.build("low")
    assert m.is_built


def test_build_medium_bvh():
    m = mesh_plane()
    m.build("medium")
    assert m.is_built


def test_build_low_bvh():
    m = mesh_plane()
    m.build("high")
    assert m.is_built


def test_build_undef_bvh():
    m = mesh_plane()
    with pytest.raises(ValueError):
        m.build("undef")


def test_single_ray_intersection():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [0.6, 0.7, 1]
    ray_direction = [0, 0, -1]
    result = m.intersect(ray_origin, ray_direction)

    assert len(result.coords) == 1
    assert result.coords[0][0] == 0.6
    assert result.coords[0][1] == 0.7
    assert result.coords[0][2] == 0

    assert len(result.tri_ids) == 1
    assert result.tri_ids[0] == 1

    assert len(result.distances) == 1
    assert result.distances[0] == 1


def test_tfar_intersection_miss():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [0.5, 0.5, 1]
    ray_direction = [0, 0, -1]
    result = m.intersect(ray_origin, ray_direction, tfar=0.5)
    assert len(result) == 1
    assert np.isnan(result.coords[0][0])


def test_tnear_intersection_miss():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [0.5, 0.5, 1]
    ray_direction = [0, 0, -1]
    result = m.intersect(ray_origin, ray_direction, tnear=2)
    assert len(result) == 1
    assert np.isnan(result.coords[0][0])


def test_occulsion():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [0.5, 0.5, 1]
    ray_direction = [0, 0, -1]
    occ = m.occlusion(ray_origin, ray_direction)
    assert len(occ) == 1
    assert occ[0] == True


def test_occulsion_miss():
    m = mesh_plane()
    m.build("medium")
    ray_origin = [0.5, 0.5, 1]
    ray_direction = [0, 0, 1]
    occ = m.occlusion(ray_origin, ray_direction)
    assert len(occ) == 1
    assert occ[0] == False
