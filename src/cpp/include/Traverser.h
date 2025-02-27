//
// Created by Dag Wästberg on 2025-02-27.
//

#ifndef TRAVERSER_H
#define TRAVERSER_H

#include "types.h"
#include "Accel.h"

// Helper class to implement generator for BVH traversal
class BvhTraverser {
public:
    bool finished;

    BvhTraverser(const Accel &accel, const Ray &ray)
        : accel_(accel), ray_(ray), finished(false) {
        // Initialize the stack with the root node
        stack_.push(accel_.bvh.get_root().index);
    }

    // Get the next triangle ID along the ray, or -1 if done
    int64_t next() {
        if (finished) return -1;

        while (!stack_.is_empty()) {
            auto top = stack_.pop();

            if (top.is_leaf()) {
                // Process leaf node
                for (size_t i = top.first_id(); i < top.first_id() + top.prim_count(); ++i) {
                    size_t tri_id = accel_.permutation_map[i];
                    current_leaf_tris_.push_back(tri_id);
                }

                // If we have triangles from this leaf, return the first one
                if (!current_leaf_tris_.empty()) {
                    int64_t result = current_leaf_tris_.back();
                    current_leaf_tris_.pop_back();
                    return result;
                }
            } else {
                // Process inner node
                auto& left = accel_.bvh.nodes[top.first_id()];
                auto& right = accel_.bvh.nodes[top.first_id() + 1];

                auto inv_dir = ray_.template get_inv_dir<true>();
                auto inv_org = -inv_dir * ray_.org;
                auto inv_dir_pad = ray_.pad_inv_dir(inv_dir);
                auto octant = ray_.get_octant();

                auto intr_left = left.intersect_robust(ray_, inv_dir, inv_dir_pad, octant);
                auto intr_right = right.intersect_robust(ray_, inv_dir, inv_dir_pad, octant);

                bool hit_left = intr_left.first <= intr_left.second;
                bool hit_right = intr_right.first <= intr_right.second;

                // Push nodes in far-to-near order (so we pop near-to-far)
                if (hit_left && hit_right) {
                    if (intr_left.first > intr_right.first) {
                        stack_.push(left.index);
                        stack_.push(right.index);
                    } else {
                        stack_.push(right.index);
                        stack_.push(left.index);
                    }
                } else {
                    if (hit_right) stack_.push(right.index);
                    if (hit_left) stack_.push(left.index);
                }
            }

            // If we have triangles from previous leaves, return the next one
            if (!current_leaf_tris_.empty()) {
                const int64_t result = current_leaf_tris_.back();
                current_leaf_tris_.pop_back();
                return result;
            }
        }

        // return the last triangles
        if (!current_leaf_tris_.empty()) {
            const int64_t result = current_leaf_tris_.back();
            current_leaf_tris_.pop_back();
            return result;
        }

        // No more triangles
        finished = true;
        return -1;
    }

private:
    const Accel &accel_;
    Ray ray_;
    bvh::v2::SmallStack<Bvh::Index, 64> stack_;
    std::vector<int64_t> current_leaf_tris_;
};



#endif //TRAVERSER_H
