//
// Created by Dag Wästberg on 2024-10-03.
//

#ifndef PYMESHRAY_UTILS_H
#define PYMESHRAY_UTILS_H

#include "types.h"
#include "Accel.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

namespace nb = nanobind;


inline Vec3 triangle_normal(const Tri &tri) {
    Vec3 e1 = tri.p1 - tri.p0;
    Vec3 e2 = tri.p2 - tri.p0;
    Vec3 n = cross(e1, e2);
    Vec3 n_norm = normalize(n);
    return n_norm;
}

inline Vec3 reflection_dir(const Vec3 &dir, const Vec3 &normal) {
    auto ndir = normalize(dir);
    return ndir - 2 * dot(ndir, normal) * normal;
}

inline Vec3 hit_reflection (const Tri &tri, const Vec3 &hit_dir) {
    Vec3 normal = triangle_normal(tri);
    Vec3 reflection = reflection_dir(hit_dir, normal);
    reflection = normalize(reflection);
    return reflection;
}

template<bool IsAnyHit, bool UseRobustTraversal>
static size_t intersect_accel(Ray &ray, const Accel &accel) {
    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
    size_t prim_id = invalid_id;
    static constexpr size_t stack_size = 64;
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    accel.bvh.intersect<IsAnyHit, UseRobustTraversal>(ray, accel.bvh.get_root().index, stack,
                                                      [&](size_t begin, size_t end) {

                                                          for (size_t i = begin; i < end; ++i) {

                                                              if (accel.precomputed_tris[i].intersect(ray))
                                                                  prim_id = i;
                                                          }
                                                          return prim_id != invalid_id;
                                                      }
    );
    return prim_id;
}

static size_t interesection_counter(Ray &ray, const Accel &accel) {
    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
    size_t prim_id = invalid_id;
    static constexpr size_t stack_size = 64;
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    size_t count = 0;

    accel.bvh.intersect<false, true>(ray, accel.bvh.get_root().index, stack,
                                    [&](size_t begin, size_t end) {
                                        for (size_t i = begin; i < end; ++i) {
                                            auto tfar_orig = ray.tmax;
                                            if (accel.precomputed_tris[i].intersect(ray)) {
                                                // Reset tmax so that we still intersect triangles further away
                                                ray.tmax = tfar_orig;
                                                count++;
                                            }
                                        }
                                        // pretend we always miss so we keep traversing
                                        return false;

                                    }
    );
    return count;
}

std::vector<Ray> pack_rays(const nb::ndarray<Scalar, nb::shape<-1, 3>> &origins,
                           const nb::ndarray<Scalar, nb::shape<-1, 3>> &directions, const nb::ndarray<Scalar, nb::ndim<1>> &tnear,
                           const nb::ndarray<Scalar, nb::ndim<1>> &tfar) {
    size_t num_rays = origins.shape(0);
    std::vector<Ray> rays;
    rays.reserve(num_rays);
    for (size_t i = 0; i < num_rays; i++) {
        rays.emplace_back(
                Vec3(origins(i, 0), origins(i, 1), origins(i, 2)),
                Vec3(directions(i, 0), directions(i, 1), directions(i, 2)),
                tnear(i), tfar(i)
        );
    }
    return rays;
}

#endif //PYMESHRAY_UTILS_H
