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

#ifndef __BVHNODE_H__
#define __BVHNODE_H__

#include "aabb.h"
#include <cstdint>


namespace lbvh {

typedef struct Node* NodePtr;
typedef struct InternalNode* InternalNodePtr;
typedef struct LeafNode* LeafNodePtr;


struct Node {
    __device__
    Node() : childA(nullptr), childB(nullptr), parent(nullptr), updateFlag(0), isLeaf(false) {}

    NodePtr childA;
    NodePtr childB;
    NodePtr parent;
    int updateFlag;
    bool isLeaf;
    AABB bbox;
};



struct InternalNode : public Node {
    using Node::Node; // Inheriting Constructor

    __device__
    InternalNode() {
        this->isLeaf = false;
    }

};


struct LeafNode : public Node {
    using Node::Node; // Inheriting Constructor

    __device__
    LeafNode() : objectID(0) {
        this->isLeaf = true;
    }
    
    std::uint32_t objectID;

};

}

#endif
