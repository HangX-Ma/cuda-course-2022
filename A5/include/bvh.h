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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace lbvh {

class BVH {

public:
    BVH() {};
    ~BVH();

    __host__ void 
    construct();

    __host__ void 
    loadObj(std::string& inputfile);

    __host__ triangle_t*
    getTriangleList() {
        return triangle_indices_h_.data();
    }

    __host__ vec3f*
    getVerticeList() {
        return vertices_h_.data();
    }

    __host__ std::uint32_t
    getOjbectNum() {
        return static_cast<std::uint32_t>(triangle_indices_h_.size());
    }

private:
    std::uint32_t* mortonCodes;
    std::uint32_t* objectIDs;

    InternalNodePtr internalNodes;  //!< num_objects - 1
    LeafNodePtr leafNodes;          //!< num_objects

    std::vector<triangle_t> triangle_indices_h_;
    std::vector<vec3f> vertices_h_;
    std::vector<vec3f> normals_h_;

    triangle_t* triangle_indices_d_;
    vec3f* vertices_d_;
    vec3f* normals_d_;
    AABB* aabbs;

};


}
#endif