# pyraymesh

## Description

`pyraymesh` is a Python library for performing ray intersection and occlusion
tests on 3D meshes using a Bounding Volume Hierarchy (BVH). The library uses
the C++ library [bvh](https://github.com/madmann91/bvh) for building the BVH and performing the intersection tests.

While this library is reasonably fast for simpler meshes (benchmarks coming soon), it is not as fast as Embree, espcially for larger and more complex meshes. However, it does not
have any dependencies on external libraries, and is thus easier to install and use.

## Installation

Install the package either by 

```sh
pip install pyraymesh
```

or cloning the repo and using pip:

```sh   
pip install .
```

Note that the package requires a C++ compiler to build the C++ extension.

## Usage

### Building the BVH

To build the BVH for a mesh:

```python
from pyraymesh import Mesh
import numpy as np

vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2], [2, 3, 0]])
mesh = Mesh(vertices, faces)
mesh.build("medium")
```

The `build` method takes a string argument that specifies the BVH build type, which can be one of the following:
"low", "medium" and "high". The build type determines the trade-off between build time and query time. For most cases
"medium" is almost always the right choice.

### Ray Intersection

To perform ray intersection tests:

```python
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [[0, 0, -1], [0, 0, 1]]
## or 
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [0, 0, -1]  # multiple rays with same direction
## or 
ray_origin = [0.1, 0.2, 1]
ray_direction = [[0, 0, -1], [0, 0, 1]]  # multiple rays with same origin

result = mesh.intersect(ray_origin, ray_direction, tnear=0, tfar=1000)
print(result.num_hits)
print(result.coords)
print(result.tri_ids)
print(result.distances)
```

`tnear` and `tfar` can be scalars or lists of the same length as the number of rays. If they are scalars, the same
value will be used for all rays. If they are lists, each value will be used for the corresponding ray.

If you set `tnear` to a value greater than 0, the intersection tests will ignore any intersections that are closer 
than `tnear`. Similarly, if you set `tfar` to a value less than infinity, the intersection tests will ignore any 
intersections that are farther than `tfar`. This library does not support negative values for `tnear` or `tfar`.

### Reflections

If you want to get the reflection of the rays, add the `calculate_reflections = True` 
parameter to the `intersect` method:

```python
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [[0, 0, -1], [0, 0, 1]]
result = mesh.intersect(ray_origin, ray_direction, tnear=0, tfar=1000, calculate_reflections=True)
print(result.reflections)
```
results.reflections is a list  of noramlized vectors representing the directions of the 
reflection of the rays. Only do this if you need the reflections, as it will slow down the
intersection tests.



### Occlusion Test

If you just care about whether a ray is occluded or not (i.e., you don't care about
the intersection point) you can use the `occlusion` method which is faster than the
`intersect` method and just returns an array of booleans.

```python
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [[0, 0, -1], [0, 0, 1]]
occluded = mesh.occlusion(ray_origin, ray_direction)
print(occluded)
```

### Count intersections

If you want to know the total number of intersections for each ray along its path, without stopping at the first
intersection, you can use the `count_intersections` method:

```python
total_intersections = mesh.count_intersections(ray_origin, ray_direction)
print(total_intersections)
```
This method returns an array of integers representing the total number of triangles that each ray intersects 
between `tnear` and `tfar`.

### Test line-of-sight

If you want to know if two points are visible to each other, you can use the `line_of_sight` method:

```python

origin_point = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
target_point = [[0, 0, -1], [0, 0, 1]]
## or 
origin_point = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
target_point = [0, 0, -1]  # multiple origin points with same target
## or 
origin_point = [0.1, 0.2, 1]
target_point = [[0, 0, -1], [0, 0, 1]]  # multiple target points with same origin

visible = mesh.line_of_sight(origin_point, target_point)
```
 `visible` is a list of booleans representing whether the target point is visible from the origin point.

### Traverse the BVH

If you want to traverse the BVH and get all triangles that are along a ray in the BVH, you can use the `traverse` method. This is useful if you want to
do some custom processing on the triangles that are potentially intersected by a ray. 

```python
origin = [0, 0, 10]
direction = [0, 0, -1]

for t_id in mesh.traverse(origin, direction):
    print(f"Triangle {mesh.vertices[mesh.faces[t_id]]} is potentially intersected by the ray.")
```

Note that the current implementation traverses the entire BVH when the method is called, even if you break early from the loop. For huge
meshes, this can be a performance bottleneck. Hopefully, this will be fixed in future versions.


### Parallelization

The `intersect` and `occlusion` methods can be parallelized by passing `threads` parameter when calling the methods:

```python
result = mesh.intersect(ray_origin, ray_direction, tnear=0, tfar=1000, threads=4)
```

The `threads` parameter specifies the number of threads to use for the intersection tests. If set to `-1`, 
the number of threads will be equal to the number of cores on the machine. In general you shouldn't set the number of 
threads to be greater than the number of cores on the machine.

For a small number of rays, the overhead of parallelization might make the parallel version slower than the serial
version, so it is recommended to test the performance of both versions for your specific use case.


## Testing

To run the tests:

```sh
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
