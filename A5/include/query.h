/**
 * @file query.h
 * @author HangX-Ma m-contour@qq.com
 * @brief lbvh tree query
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

#ifndef __QUERY_H__
#define __QUERY_H__

#include "aabb.h"
#include "bvh_node.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace lbvh {


// query object indices that potentially overlaps with query aabb.
__device__ std::uint32_t 
query_device(AABB* target, Node* internalNodes, 
    std::uint32_t* outBuffer, bool ifFirst, const std::uint32_t max_buffer_size = 0xFFFFFFFF) {

    Node* stack[64];
    Node** stackPtr = stack;
    // root node is always zero
    *stackPtr++ = &internalNodes[0];

    std::uint32_t found_num = 0;
    do {
        Node* node = *--stackPtr;
        Node* leftNode   = node->leftChild;
        Node* rightNode  = node->rightChild;

        // check left child tree
        if (overlaps(*target, leftNode->bbox)) {
            if (leftNode->isLeaf) {
                if ((!ifFirst) && found_num < (max_buffer_size)) {
                    outBuffer[found_num] = (leftNode)->objectID;
                }
                ++found_num;
            } 
            else {
                *stackPtr++ = leftNode;
            }
        }

        // check right child tree
        if (overlaps(*target, rightNode->bbox)) {
            if (rightNode->isLeaf) {
                if (((!ifFirst)) && found_num < (max_buffer_size)) {
                    outBuffer[found_num] = (rightNode)->objectID;
                }
                ++found_num;
            } 
            else {
                *stackPtr++ = rightNode;
            }
        }
    } while ( stack < stackPtr);

    return found_num;
}

}
#endif