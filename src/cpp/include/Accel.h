//
// Created by Dag Wästberg on 2024-10-03.
//

#ifndef PYMESHRAY_ACCEL_H
#define PYMESHRAY_ACCEL_H

#include "types.h"

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include <iostream>

class Accel {
public:
    Bvh bvh;
    std::vector<PrecomputedTri> precomputed_tris;
    std::vector<size_t> permutation_map;

    Accel(const std::vector<Tri> &tris, const std::string &quality) {
        bvh::v2::ThreadPool thread_pool;
        bvh::v2::ParallelExecutor executor(thread_pool);

        permutation_map.resize(tris.size());

        std::vector<BBox> bboxes(tris.size());
        std::vector<Vec3> centers(tris.size());
        executor.for_each(0, tris.size(), [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                bboxes[i] = tris[i].get_bbox();
                centers[i] = tris[i].get_center();
            }
        });

        typename bvh::v2::DefaultBuilder<Node>::Config config;
        if (quality == "high") {
            config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
        } else if (quality == "medium") {
            config.quality = bvh::v2::DefaultBuilder<Node>::Quality::Medium;
        } else if (quality == "low") {
            config.quality = bvh::v2::DefaultBuilder<Node>::Quality::Low;
        } else {
            std::cerr << "Unknown quality level: " << quality << ", using medius" << std::endl;
            config.quality = bvh::v2::DefaultBuilder<Node>::Quality::Medium;
        }
        bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);

        precomputed_tris.resize(tris.size());
        // This precomputes some data to speed up traversal further.
        executor.for_each(0, tris.size(), [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto j = bvh.prim_ids[i];
                precomputed_tris[i] = tris[j];
                permutation_map[j] = i;
            }
        });
    }
};

#endif //PYMESHRAY_ACCEL_H
