/**
 * @file bvh.cuh
 * @author HangX-Ma m-contour@qq.com
 * @brief  BVH tree class
 * @version 0.1
 * @date 2022-10-10
 * 
 * @copyright Copyright (c) 2022 HangX-Ma(MContour) m-contour@qq.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __BVH_H__
#define __BVH_H__

#include "bvh_node.h"
#include "morton_code.h"
#include "triangle.h"
#include "query.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace lbvh {

enum class BVH_STATUS {
    STATE_INITIAL,
    STATE_CONSTRUCT,
    STATE_GET_NEIGHBOUR,
};


typedef struct bvh_device bvh_device;

struct bvh_device {
    std::uint32_t num_nodes;
    std::uint32_t num_objects;
    
    InternalNodePtr internalNodes;  //!< num_objects - 1
    LeafNodePtr leafNodes;          //!< num_objects
    AABB* aabbs_d_;
    std::uint32_t* objectIDs;
};


class BVH {

public:
    __host__ __inline__ static BVH*
    getInstance() {
        static BVH bvh;
        return &bvh;
    }

    __host__ void 
    construct();

    __host__ void 
    loadObj(std::string& inputfile);

    __host__ triangle_t*
    getTriangleList() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL) {
            return triangle_indices_h_.data();
        }
        return nullptr;
    }

    __host__ vec3f*
    getVerticeList() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL) {
            return vertices_h_.data();
        }
        return nullptr;
    }

    __host__ std::uint32_t
    getOjbectNum() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL) {
            return static_cast<std::uint32_t>(triangle_indices_h_.size());
        }
        return NULL;
    }
    
    __host__ bvh_device 
    getDevPtrs() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL) {
            return bvh_device{ 2 * num_objects - 1, num_objects, 
                            internalNodes, leafNodes, aabbs_d_, objectIDs};
        }
        return bvh_device{NULL, NULL, nullptr, nullptr, nullptr, nullptr};
    }

    /**
     * @brief Get the neighbour information
     * @note adjInfo_d_ store the adjacent objectIDs
     */
    __device__ void 
    getNbInfo() {
        adjInfo_d_.clear();
        adjInfo_d_.resize(num_objects);
        thrust::for_each(thrust::device,
            thrust::make_counting_iterator<std::size_t>(0),
            thrust::make_counting_iterator<std::size_t>(num_objects),
            [this] __device__ (std::uint32_t idx) {
                lbvh::query_device(aabbs_d_ + idx, aabbs_d_, internalNodes, adjInfo_d_.data() + idx, 20);
            });
    }


private:
    BVH() {};
    ~BVH();

    BVH_STATUS bvh_status = BVH_STATUS::STATE_INITIAL;

    std::uint32_t* mortonCodes;
    std::uint32_t* objectIDs;
    std::uint32_t num_objects;

    InternalNodePtr internalNodes;  //!< num_objects - 1
    LeafNodePtr leafNodes;          //!< num_objects

    std::vector<triangle_t> triangle_indices_h_;
    std::vector<vec3f> vertices_h_;
    std::vector<vec3f> normals_h_;

    triangle_t* triangle_indices_d_;
    vec3f* vertices_d_;
    vec3f* normals_d_;
    AABB* aabbs_d_;
    
    thrust::device_vector<thrust::device_vector<std::uint32_t>> adjInfo_d_;
};


}
#endif