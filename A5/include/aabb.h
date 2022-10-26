/**
 * @file aabb.h
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

#ifndef __AABB_H__
#define __AABB_H__

#include "vector3f.h"
#include <float.h>

namespace lbvh {
class AABB {

public:
    vec3f bmin;
    vec3f bmax;

    __host__ __device__
    AABB() {
        bmax = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        bmin = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
    }

    __host__ __device__
    AABB(vec3f bmin, vec3f bmax) : bmin(bmin), bmax(bmax) {}

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

__host__ __device__ __inline__
AABB merge(const AABB& lhs, const AABB& rhs) {
    AABB merged;
    merged.bmax.x = fmaxf(lhs.bmax.x, rhs.bmax.x);
    merged.bmax.y = fmaxf(lhs.bmax.y, rhs.bmax.y);
    merged.bmax.z = fmaxf(lhs.bmax.z, rhs.bmax.z);
    merged.bmin.x = fminf(lhs.bmin.x, rhs.bmin.x);
    merged.bmin.y = fminf(lhs.bmin.y, rhs.bmin.y);
    merged.bmin.z = fminf(lhs.bmin.z, rhs.bmin.z);
    
    return merged;
}

__host__ __device__ __inline__ bool 
overlaps(const AABB& lhs, const AABB& rhs) {
    if (lhs.bmin.x > rhs.bmax.x) return false;
    if (lhs.bmin.y > rhs.bmax.y) return false;
    if (lhs.bmin.z > rhs.bmax.z) return false;

    if (lhs.bmax.x < rhs.bmin.x) return false;
    if (lhs.bmax.y < rhs.bmin.y) return false;
    if (lhs.bmax.z < rhs.bmin.z) return false;

    return true;
}

}



#endif 