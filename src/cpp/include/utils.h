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

std::vector<Ray> pack_rays(const nb::ndarray<Scalar, nb::shape<-1, 3>> &origins,
                           const nb::ndarray<Scalar, nb::shape<-1, 3>> &directions, Scalar tmin = 0,
                           Scalar tmax = std::numeric_limits<Scalar>::max()) {
    size_t num_rays = origins.shape(0);
    std::vector<Ray> rays;
    rays.reserve(num_rays);
    for (size_t i = 0; i < num_rays; i++) {
        rays.emplace_back(
                Vec3(origins(i, 0), origins(i, 1), origins(i, 2)),
                Vec3(directions(i, 0), directions(i, 1), directions(i, 2)),
                tmin, tmax
        );
    }
    return rays;
}

#endif //PYMESHRAY_UTILS_H
