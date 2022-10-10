/**
 * @file AABB.cuh
 * @author HangX-Ma m-contour@qq.com
 * @brief AABB bounding box struct definition 
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

#include "Vector3f.cuh"

typedef struct {
    Vector3f bmin;
    Vector3f bmax;

    __host__ __device__ Vector3f
    getCentroid () {
        return (bmin + bmax) / 2;
    }

    __host__ __device__ void
    operator=(const s_AABB &other) {
        this->bmin = other.bmin;
        this->bmax = other.bmax;
    }

} s_AABB;

#endif 