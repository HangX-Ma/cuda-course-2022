/**
 * @file bvh_node.h
 * @author HangX-Ma m-contour@qq.com
 * @brief  BVH node class
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

#ifndef __BVH_NODE_H__
#define __BVH_NODE_H__

#include "aabb.h"
#include <cstdint>


namespace lbvh {


class Node {
public:
    __device__
    Node() : leftChild(nullptr), rightChild(nullptr), parent(nullptr), 
        updateFlag(0), isLeaf(false), objectID(0) {}
    /* virtual destructor can avoid memory leak through 
    safely destroy derived class and base class in order. */
    // virtual ~Node() {};
    Node* leftChild;
    Node* rightChild;
    Node* parent;
    int updateFlag;
    bool isLeaf;
    AABB bbox;
    std::uint32_t objectID;
};

}

#endif
