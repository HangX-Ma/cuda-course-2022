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

#ifndef __BVH_CUH__
#define __BVH_CUH__

#include "bvh_node.cuh"
#include "morton_code.cuh"
#include "triangle.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace lbvh {

class BVH {

public:
    BVH() = default;
    ~BVH();

    __host__ void 
    construct(std::string inputfile);

    __host__ __device__ __inline__ AABB& 
    getBoundAABB() {
        return aabb_bound;
    }

private:
    __host__ void 
    loadObj(std::string& inputfile);

    __uint32_t* mortonCodes;
    __uint32_t* objectIDs;

    InternalNodePtr internalNodes; //!< Triangle_nums - 1
    LeafNodePtr LeafNodes; //!< Triangles_num
    
    AABB aabb_bound;

    thrust::host_vector<triangle_t> triangle_indices_h_;
    thrust::host_vector<vec3f> vertices_h_;
    thrust::host_vector<vec3f> normals_h_;

    triangle_t* triangle_indices_d_;
    vec3f* vertices_d_;
    vec3f* normals_d_;
    AABB* aabbs_d_;

};


}
#endif