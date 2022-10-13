/**
 * @file aabb.cuh
 * @author HangX-Ma m-contour@qq.com
 * @brief AABB bounding box class definition 
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

#ifndef __AABB_CUH__
#define __AABB_CUH__

#include "vector3f.cuh"
#include <float.h>

namespace lbvh {
class AABB {

public:

    AABB() { init(); }

    vec3f bmin;
    vec3f bmax;

    __host__ __device__ __inline__ void 
    init() {
        bmax = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        bmin = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
    }

    __host__ __device__ __inline__ bool 
    overlaps(const AABB& other) const {
        if (bmin.x > other.bmax.x) return false;
        if (bmin.y > other.bmax.y) return false;
        if (bmin.z > other.bmax.z) return false;

        if (bmax.x < other.bmin.x) return false;
        if (bmax.y < other.bmin.y) return false;
        if (bmax.z < other.bmin.z) return false;

        return true;
    }

    __host__ __device__ __inline__ float
    getWidth () const {
        return bmax.x - bmin.x;
    }

    __host__ __device__ __inline__ float
    getHeight () const {
        return bmax.y - bmin.y;
    }

    __host__ __device__ __inline__ float
    getDepth () const {
        return bmax.z - bmin.z;
    }

    __host__ __device__ __inline__ vec3f
    getCentroid () const {
        return (bmin + bmax) * 0.5f;
    }

    __host__ __device__ __inline__ bool 
    empty() const {
        return bmax.x < bmin.x;
    }

    __host__ __device__ __inline__ void
    operator=(const AABB &other) {
        this->bmin = other.bmin;
        this->bmax = other.bmax;
    }



};
}



#endif 