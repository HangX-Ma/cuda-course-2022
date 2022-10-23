/**
 * @file triangle.h
 * @author HangX-Ma m-contour@qq.com
 * @brief triangle struct type definition
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

#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include "vector3f.h"
#include "tiny_obj_loader.h"

namespace lbvh {
 
typedef struct tri triangle_t;

/**
 * @brief triangle struct type, a, b, c represent 
 * the angles of one triangle respectively.
 */
struct tri {
    tinyobj::index_t a;
    tinyobj::index_t b;
    tinyobj::index_t c;
};


}
#endif