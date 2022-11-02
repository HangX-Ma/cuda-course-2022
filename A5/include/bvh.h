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
#include <cuda_runtime.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <limits>

namespace lbvh {

enum class BVH_STATUS {
    STATE_INITIAL,
    STATE_CONSTRUCT,
    STATE_GET_NEIGHBOUR,
    STATE_PROPAGATE
};


typedef struct bvh_device bvh_device;

struct bvh_device {
    std::uint32_t num_nodes;
    std::uint32_t num_objects;
    
    Node* internalNodes;  //!< num_objects - 1
    Node* leafNodes;          //!< num_objects
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

    __host__ BVH_STATUS
    getStatus() {
        return bvh_status;
    }

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

    __host__ vec3f*
    getNormalList() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL) {
            return normals_h_.data();
        }
        return nullptr;
    }

    __host__ std::uint32_t*
    getSortedObjectIDs() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL &&
            bvh_status != BVH_STATUS::STATE_CONSTRUCT) {
            return sortedObjectIDs_h_.data();
        }
        return nullptr;
    }

    __host__ std::uint32_t
    getOjbectNum() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL) {
            return static_cast<std::uint32_t>(triangle_indices_h_.size());
        }
        return 0xFFFFFFFF;
    }
    
    __host__ std::uint32_t
    getAdjObjectNum() {
        if (bvh_status == BVH_STATUS::STATE_PROPAGATE) {
            return num_adjObjects;
        }
        return 0xFFFFFFFF;
    }

    __host__ bvh_device 
    getDevPtrs() {
        if (bvh_status != BVH_STATUS::STATE_INITIAL) {
            return bvh_device{ 2 * num_objects - 1, num_objects, internalNodes, leafNodes, 
                    aabbs_d_, thrust::raw_pointer_cast(sortedObjectIDs_d_.data())};
        }
        return bvh_device{0xFFFFFFFF, 0xFFFFFFFF, nullptr, nullptr, nullptr, nullptr};
    }

    /**
     * @brief Get the neighbour information
     * @note adjInfo_d_ store the adjacent objectIDs
     */
    __host__ void 
    getNbInfo();
    
    
    /**
     * @brief Method to call propagating process kernel
     */
    __host__ void 
    propagate();

private:
    BVH() : mortonCodes(nullptr), num_objects(0), num_adjObjects(0) {};
    ~BVH();

    BVH_STATUS bvh_status = BVH_STATUS::STATE_INITIAL;

    std::uint32_t* mortonCodes;
    thrust::device_vector<std::uint32_t> sortedObjectIDs_d_;
    thrust::host_vector<std::uint32_t> sortedObjectIDs_h_;

    std::uint32_t num_objects;
    std::uint32_t num_adjObjects;

    Node* internalNodes;  //!< num_objects - 1
    Node* leafNodes;          //!< num_objects

    thrust::host_vector<triangle_t> triangle_indices_h_;
    thrust::host_vector<vec3f> vertices_h_;
    thrust::host_vector<vec3f> normals_h_;

    triangle_t* triangle_indices_d_;
    vec3f* vertices_d_;
    AABB* aabbs_d_;
    
    /* store each object's neighbour's ids */
    std::uint32_t* adjObjInfo_d_;

    /* exclusive scan (prefix sum) result container */
    thrust::device_vector<std::uint32_t> scan_res_d_;
    thrust::host_vector<std::uint32_t> scan_res_h_;
    
    /* store each object's neighbour number */
    thrust::device_vector<std::uint32_t> adjObjNumList_d_;
    thrust::host_vector<std::uint32_t> adjObjNumList_h_;
};


}
#endif