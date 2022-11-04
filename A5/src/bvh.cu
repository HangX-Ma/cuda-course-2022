#include "tiny_obj_loader.h"

#include "common.h"
#include "bvh.h"
#include "morton_code.h"
#include "query.h"

#include <device_launch_parameters.h>

#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <sys/stat.h> // check file status

// #define DEBUG_INFO 1


static std::string div_signs(10, '-');


namespace lbvh {

/* get common upper bits */
__device__ __inline__ int 
commonUpperBits(const unsigned int lhs, const unsigned int rhs) noexcept {
    return __clz(lhs ^ rhs);
}

/* get cuda device info */
void cudaDevInfo();

/* Func declaration */
__device__ int2 
determineRange(std::uint32_t* sortedMortonCodes, int num_objects, int idx);

__device__ __inline__ int
findSplit(std::uint32_t* sortedMortonCodes, int first, int last);

/* Kernel declaration */
__global__ void 
computeBBoxes_kernel(const std::uint32_t num_objects, 
        triangle_t* trianglePtr, vec3f* verticePtr, AABB* aabbPtr);

__global__ void 
computeMortonCode_kernel(std::uint32_t num_objects, std::uint32_t* objectIDs, 
        AABB aabb_bound, AABB* aabbs, std::uint32_t* mortonCodes);

__global__ void
construtInternalNodes_kernel(std::uint32_t* sortedMortonCodes, std::uint32_t* sortedObjectIDs, 
        int numObjects, Node* internalNodes, Node* leafNodes, AABB* bboxes);

__global__ void
createAABBHierarchy_Kernel(int num_objects, Node* leafNodes, Node* internalNodes);

__global__ void 
NbInfoScan_Kernel_First(std::uint32_t num_object, AABB* aabbs, Node* internalNodes, 
        std::uint32_t* adjObjNums, std::uint32_t* adjObjectsOut, std::uint32_t* prefix_sum);

__global__ void 
NbInfoScan_Kernel_Second(std::uint32_t num_object, AABB* aabbs, Node* internalNodes, 
        std::uint32_t* adjObjNums, std::uint32_t* adjObjectsOut, std::uint32_t* prefix_sum);

struct minUnaryFunc{
    __host__ __device__
    vec3f operator () (const AABB& a){
        return a.bmin;
    }
};

struct minBinaryFunc{
    __host__ __device__
    vec3f operator () (const vec3f& a, const vec3f& b){
        return vmin(a,b);
    }
};
struct maxUnaryFunc{
    
    __host__ __device__
    vec3f operator () (const AABB& a){
        return a.bmax;
    }
};

struct maxBinaryFunc{
    __host__ __device__
    vec3f operator () (const vec3f& a, const vec3f& b){
        return vmax(a,b);
    }
};


__host__ void 
BVH::loadObj(std::string& inputfile) {
    struct stat buffer;  
    if (stat(inputfile.c_str(), &buffer) != 0) {
        printf("No file found according to given argument <%s>.\n", inputfile.c_str());
        exit(EXIT_FAILURE);
    }


    std::string suffixStr = inputfile.substr(inputfile.find_last_of('.') + 1);
    if (suffixStr != std::string("obj")) {
        printf("Invalid file type. Please select .obj suffix file.\n");
        exit(EXIT_FAILURE);
    }

    /* stage info */
    std::cout << div_signs << "  Stage 1: Loading objects  " << div_signs << std::endl;
    printf("Loading objcts from <%s> ...\n", inputfile.c_str());

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, inputfile.c_str());

    /* deal with unexpected situation */
    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(EXIT_FAILURE);
    }

    // Loop over shapes
    size_t shapes_size = shapes.size();
    for (size_t s = 0; s < shapes_size; s++) {
        // Loop over faces (polygon)
        size_t index_offset = 0;
        size_t face_num = shapes[s].mesh.num_face_vertices.size(); // total face number
        for (size_t f = 0; f < face_num; f++) {
            /* store triangle's three vertices (a,b,c) index 
            ( index_t: vertex_index, normal_index, texcoord_index) */
            triangle_t tri;
            tri.a = shapes[s].mesh.indices[index_offset + 0];
            tri.b = shapes[s].mesh.indices[index_offset + 1];
            tri.c = shapes[s].mesh.indices[index_offset + 2];
            // triangle points property
            triangle_indices_h_.push_back(tri);
            // per-face material
            index_offset += 3;
        }
    }

    /* get vertices */
    size_t vertices_size = attrib.vertices.size();
    for (size_t s = 0; s < vertices_size; s += 3) {
        vec3f vertice;
        vertice.x = attrib.vertices.at(s + 0);
        vertice.y = attrib.vertices.at(s + 1);
        vertice.z = attrib.vertices.at(s + 2);
        vertices_h_.push_back(vertice);
    }

    /* get normals */
    size_t normals_size = attrib.normals.size();
    for (size_t s = 0; s < normals_size; s += 3) {
        vec3f normal;
        normal.x = attrib.normals.at(s + 0);
        normal.y = attrib.normals.at(s + 1);
        normal.z = attrib.normals.at(s + 2);
        normals_h_.push_back(normal);
    }
    
    printf("objcet size: %lu, vertices size: %lu, normals size: %lu.\n", 
                        triangle_indices_h_.size(), vertices_h_.size(), normals_h_.size());

    bvh_status = BVH_STATUS::STATE_CONSTRUCT;

    return;
}


__host__ void 
BVH::construct() {
    if (bvh_status != BVH_STATUS::STATE_CONSTRUCT) {
        printf("Please do construction in order. \
            Error happens in %s at line %d.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if(triangle_indices_h_.size() == 0u || 
        vertices_h_.size() == 0u || 
        normals_h_.size() == 0u ) {

        printf("Please load objects fisrt. Error happens in %s at line %d.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    std::cout << div_signs << "  Start LBVH Construction  " << div_signs << std::endl;
    printf("[CUDA device information]\n");
    cudaDevInfo();

    /* basic information */
                        num_objects        = triangle_indices_h_.size();
    const std::uint32_t num_internal_nodes = num_objects - 1;
    const std::uint32_t num_nodes          = num_objects * 2 - 1;
    /* kernel property */
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_objects + threadsPerBlock - 1) / threadsPerBlock;

    /* stage info assistant  */
    printf("[Kernel property]\n block:              %d\n threads:            %d\n\n", 
                blocksPerGrid, threadsPerBlock);
    printf("[LBVH property]\n leaf nodes:         %u\n internal nodes:     %u\n totol nodes:        %u\n", 
                num_objects, num_internal_nodes, num_nodes);
    printf(" num_objects size:   %lu bytes\n aabbs size:         %lu bytes\n vertices size:      %lu bytes\n\n", 
                num_objects * sizeof(triangle_t), num_objects * sizeof(AABB), normals_h_.size() * sizeof(vec3f));

    TIMING_BEGIN
    /* ---------------- STAGE 1: load objects ---------------- */
    /* allocte specific memory size */
    HANDLE_ERROR(cudaMalloc((void**)&triangle_indices_d_, num_objects * sizeof(triangle_t)));
    HANDLE_ERROR(cudaMalloc((void**)&vertices_d_, vertices_h_.size() * sizeof(vec3f)));
    HANDLE_ERROR(cudaMalloc((void**)&aabbs_d_, num_objects * sizeof(AABB)));


    /* copy data from host to device */
    HANDLE_ERROR(cudaMemcpy(triangle_indices_d_, triangle_indices_h_.data(), 
                            num_objects * sizeof(triangle_t), 
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(vertices_d_, vertices_h_.data(), 
                            vertices_h_.size() * sizeof(vec3f), 
                            cudaMemcpyHostToDevice));

    /* construct aabb */
    std::cout << div_signs << "  Stage 2: Compute AABB bounding boxes.  " << div_signs << std::endl;
    computeBBoxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(num_objects, triangle_indices_d_, vertices_d_, aabbs_d_);

    /* calculate morton code for all objects */
    AABB aabb_bound;
    thrust::device_ptr<AABB> aabb_d_ptr(aabbs_d_);
    aabb_bound.bmax = thrust::transform_reduce(
        aabb_d_ptr, aabb_d_ptr + num_objects,
        maxUnaryFunc(),
        vec3f(-1e9f, -1e9f, -1e9f),
        maxBinaryFunc());

    aabb_bound.bmin = thrust::transform_reduce(
        aabb_d_ptr, aabb_d_ptr + num_objects,
        minUnaryFunc(),
        vec3f(1e9f, 1e9f, 1e9f),
        minBinaryFunc());

    printf("--> found AABB bound min(%0.6f, %0.6f, %0.6f)\n" , aabb_bound.bmin.x , aabb_bound.bmin.y , aabb_bound.bmin.z);
    printf("--> found AABB bound max(%0.6f, %0.6f, %0.6f)\n" , aabb_bound.bmax.x , aabb_bound.bmax.y , aabb_bound.bmax.z);

    /* ---------------- STAGE 2: build BVH Tree ---------------- */
    HANDLE_ERROR(cudaMalloc(&mortonCodes, num_objects * sizeof(std::uint32_t)));
    HANDLE_ERROR(cudaMalloc(&leafNodes, num_objects * sizeof(Node)));
    HANDLE_ERROR(cudaMalloc(&internalNodes, (num_objects - 1) * sizeof(Node)));
    sortedObjectIDs_d_.resize(num_objects);
    std::uint32_t* sortedObjectIDs_d_rawPtr = thrust::raw_pointer_cast(sortedObjectIDs_d_.data());

    /* compute morton code */
    std::cout << div_signs << "  Stage 3: Calculate morton codes.  " << div_signs << std::endl;
    computeMortonCode_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (num_objects, sortedObjectIDs_d_rawPtr, aabb_bound, aabbs_d_, mortonCodes);

    /* sort morton codes */
    thrust::device_ptr<std::uint32_t> mortonCodes_d_ptr(mortonCodes);
    thrust::device_ptr<std::uint32_t> objectIDs_d_ptr = sortedObjectIDs_d_.data();
    thrust::sort_by_key(mortonCodes_d_ptr, mortonCodes_d_ptr + num_objects, objectIDs_d_ptr);
    printf("--> morton codes have been sorted.\n");

    /* copy data from device to host */
    sortedObjectIDs_h_ = sortedObjectIDs_d_;

    std::cout << div_signs << "  Stage 4: Construct LBVH hierarchy.  " << div_signs << std::endl;
    /* construct leaf nodes */
    thrust::device_ptr<Node> leafNodes_d_ptr(leafNodes);
    thrust::transform(objectIDs_d_ptr, 
        objectIDs_d_ptr + num_objects,
        leafNodes_d_ptr, 
        [] __device__ (const std::uint32_t idx){
            Node leaf;
            leaf.isLeaf = true;
            leaf.objectID = idx;
            return leaf;
        });

    printf("--> leaf nodes have been constructed.\n");
    /* construct internal nodes */
    construtInternalNodes_kernel<<<blocksPerGrid, threadsPerBlock>>>
            (mortonCodes, sortedObjectIDs_d_rawPtr, num_objects, internalNodes, leafNodes, aabbs_d_);
    printf("--> internal nodes have been constructed.\n");

    /* create AABB for each node by bottom-up strategy */
    createAABBHierarchy_Kernel<<<blocksPerGrid, threadsPerBlock>>>(num_objects, leafNodes, internalNodes);
    printf("--> lbvh hierarchy has been constructed.\n");

    bvh_status = BVH_STATUS::STATE_GET_NEIGHBOUR;
    
    TIMING_END("--> Building lbvh cost:")

    return;
}


BVH::~BVH() {
    HANDLE_ERROR(cudaFree(adjObjInfo_d_));
    HANDLE_ERROR(cudaFree(internalNodes));
    HANDLE_ERROR(cudaFree(leafNodes));
    HANDLE_ERROR(cudaFree(mortonCodes));
    HANDLE_ERROR(cudaFree(aabbs_d_));
    HANDLE_ERROR(cudaFree(vertices_d_));
    HANDLE_ERROR(cudaFree(triangle_indices_d_));
}

__host__ void 
BVH::getNbInfo() {
    if (bvh_status != BVH_STATUS::STATE_GET_NEIGHBOUR) {
        printf("Please call this method at STATE_GET_NEIGHBOUR.\n");
        return;
    }
    std::cout << div_signs << " Stage 5: Get Triangles' adjecent neighbour." << div_signs << std::endl;

    TIMING_BEGIN
    /* before we use thrust vector, we need to allocate memory for it first. */
    adjObjNumList_d_.resize(num_objects);
    std::uint32_t* adjObjNumList_d_rawPtr = thrust::raw_pointer_cast(adjObjNumList_d_.data());

    /* kernel property */
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_objects + threadsPerBlock - 1) / threadsPerBlock;

    /* first to get object number for exlusive scan */
    NbInfoScan_Kernel_First<<<blocksPerGrid, threadsPerBlock>>>
            (num_objects, aabbs_d_, internalNodes, adjObjNumList_d_rawPtr, nullptr, nullptr);
    HANDLE_ERROR(cudaDeviceSynchronize());

    adjObjNumList_h_ = adjObjNumList_d_;
    std::uint32_t* adjObjNumList_raw_ptr = thrust::raw_pointer_cast(adjObjNumList_h_.data());
    /* get adjacent neighbour sum */
    num_adjObjects = 0;
    for (int i = 0; i < num_objects; i++) {
        num_adjObjects += adjObjNumList_raw_ptr[i];
    }
    // num_adjObjects = thrust::reduce(adjObjNumList_d_.begin(), adjObjNumList_d_.end(), 0, thrust::plus<std::uint32_t>());
    printf("--> Triangles: %u, AdjTriangles: %u\n", num_objects, num_adjObjects);

    /* adjacent neighbour info container */
    HANDLE_ERROR(cudaMalloc((void**)(&adjObjInfo_d_), num_adjObjects * sizeof(std::uint32_t)));

    /* exclusive prefix sum */
    scan_res_d_.resize(num_objects);
    thrust::exclusive_scan(thrust::device, adjObjNumList_d_.begin(), adjObjNumList_d_.end(), scan_res_d_.begin());

    /* copy data from deivce to host */
    scan_res_h_ = scan_res_d_;
    std::uint32_t* scan_res_d_rawPtr = thrust::raw_pointer_cast(scan_res_d_.data());

    /* second to get objects id and store them to right position */
    NbInfoScan_Kernel_Second<<<blocksPerGrid, threadsPerBlock>>>
            (num_objects, aabbs_d_, internalNodes, adjObjNumList_d_rawPtr, adjObjInfo_d_, scan_res_d_rawPtr);
    HANDLE_ERROR(cudaDeviceSynchronize());

    TIMING_END("--> Finding adjacent nodes cost:")
    bvh_status = BVH_STATUS::STATE_PROPAGATE;
}

/**
 * @brief In order to construct a binary radix tree, we need to determine 
 * the range of keys covered by each internal node, as well as its children. 
 * 
 * @param sortedMortonCodes morton codes which have been sorted
 * @param num_objects leaf node number
 * @param idx thread or object ID
 * @return range 
 */
__device__ int2 
determineRange(std::uint32_t* sortedMortonCodes, int num_objects, int idx) {
    // determine the range of keys covered by each internal node (as well as its children)
    // direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    // the index is either the beginning of the range or the end of the range
    if (idx == 0) {
        return make_int2(0, num_objects - 1);
    } // When Idx = 0, this means the range cover the whole array

    int direction = 0;
    int commonPrefix_L = 0;
    int commonPrefix_R = 0;

    /* get current key_idx neighbors' common prefixes and then determine the direction
    so that we can get the minimum common prefix according to direction */
    commonPrefix_R = commonUpperBits(sortedMortonCodes[idx], sortedMortonCodes[idx+1]);
    commonPrefix_L = commonUpperBits(sortedMortonCodes[idx], sortedMortonCodes[idx-1]);
    direction = (commonPrefix_L - commonPrefix_R) > 0 ? -1 : 1;

    int commonPrefix_min = commonUpperBits(sortedMortonCodes[idx], sortedMortonCodes[idx - direction]);

    /* find the upper bound roughly, exponentially increasing step until the condition is violiated. */
    int lmax = 2;
    int largerCommonPrefixDir_index = idx + lmax * direction;
    while ( largerCommonPrefixDir_index >= 0 && 
            largerCommonPrefixDir_index < num_objects && 
            commonUpperBits(sortedMortonCodes[idx], sortedMortonCodes[largerCommonPrefixDir_index]) > commonPrefix_min) {
        lmax *= 2;
        largerCommonPrefixDir_index = idx + lmax * direction;
    }

    /* find the other end using binary search, this will get a preciser bound */
    int l = 0;
    int t = lmax >> 1;
    while (t > 0) {
        largerCommonPrefixDir_index = idx + (l + t) * direction;
        if (largerCommonPrefixDir_index >= 0 &&
            largerCommonPrefixDir_index < num_objects &&
            commonUpperBits(sortedMortonCodes[idx], sortedMortonCodes[largerCommonPrefixDir_index]) > commonPrefix_min) {
                l = l + t;
        }
        t = t >> 1;
    }
    /* precise upper bound index */
    std::uint32_t jdx = idx + l * direction;

    /* make sure that idx < jdx */
    if (direction < 0) {
        thrust::swap(idx, jdx);
    }

    return make_int2(idx, jdx);
}



__device__ __inline__ int
findSplit(std::uint32_t* sortedMortonCodes, int first, int last) {
    // Identical Morton codes => split the range in the middle.
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode) {
        return (first + last) >> 1;
    }

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = commonUpperBits(firstCode, lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first; // initial guess
    int step = last - first;

    do {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last) {
            std::uint32_t splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = commonUpperBits(firstCode, splitCode);
            if (splitPrefix > commonPrefix) {
                split = newSplit; // accept proposal
            }
        }
    }
    while (step > 1);

    return split;
}


__global__ void 
computeBBoxes_kernel(const std::uint32_t num_objects, triangle_t* triangles, vec3f* vertices, AABB* aabbs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > num_objects - 1) {
        return;
    } // leaf node index range [0, n - 1]

    vec3f triA_vec = vertices[triangles[idx].a.vertex_index];
    vec3f triB_vec = vertices[triangles[idx].b.vertex_index];
    vec3f triC_vec = vertices[triangles[idx].c.vertex_index];

    aabbs[idx].bmax = vmax(triA_vec, triB_vec);
    aabbs[idx].bmin = vmin(triA_vec, triB_vec);
    aabbs[idx].bmax = vmax(triC_vec, aabbs[idx].bmax);
    aabbs[idx].bmin = vmin(triC_vec, aabbs[idx].bmin);
}



__global__ void 
computeMortonCode_kernel(std::uint32_t num_objects, std::uint32_t* objectIDs, 
                            AABB aabb_bound, AABB* aabbs, std::uint32_t* mortonCodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_objects - 1) {
        return;
    } // leaf node index range [0, n - 1]

    objectIDs[idx] = idx;
    vec3f centroid = aabbs[idx].getCentroid();
    centroid.x = (centroid.x - aabb_bound.bmin.x) / (aabb_bound.bmax.x - aabb_bound.bmin.x);
    centroid.y = (centroid.y - aabb_bound.bmin.y) / (aabb_bound.bmax.y - aabb_bound.bmin.y);
    centroid.z = (centroid.z - aabb_bound.bmin.z) / (aabb_bound.bmax.z - aabb_bound.bmin.z);
    mortonCodes[idx] = morton3D(centroid);

    return;
}

__global__ void
construtInternalNodes_kernel(std::uint32_t* sortedMortonCodes, std::uint32_t* sortedObjectIDs, int numObjects,
                            Node* internalNodes, Node* leafNodes, AABB* bboxes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > numObjects - 2) {
        return;
    } // internal nodes index range [0, n - 2]
    internalNodes[idx].isLeaf = false;

    /* Find out which range of objects the node corresponds to. */
    int2 range = determineRange(sortedMortonCodes, numObjects, idx);
    int first = range.x;
    int last = range.y;

    /* Determine where to split the range. */
    int split = findSplit(sortedMortonCodes, first, last);

    // Select left child.
    Node* leftChild;
    if (split == first) {
        leftChild = &leafNodes[split];
        leftChild->bbox = bboxes[leftChild->objectID];
    } // only one node remained, so that this node must be a leaf node
    else {
        leftChild = &internalNodes[split];
    } 

    // Select right child.
    Node* rightChild;
    if (split + 1 == last) {
        rightChild = &leafNodes[split + 1];
        rightChild->bbox = bboxes[rightChild->objectID];
    }
    else {
        rightChild = &internalNodes[split + 1];
    }

    // Record parent-child relationships.
    internalNodes[idx].leftChild = leftChild;
    internalNodes[idx].rightChild = rightChild;
    leftChild->parent = &internalNodes[idx];
    rightChild->parent = &internalNodes[idx];
    
    // Node 0 is the root.
}


__global__ void
createAABBHierarchy_Kernel(int num_objects, Node* leafNodes, Node* internalNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > num_objects - 1)
        return;

    Node* nodeIdxParent = (leafNodes + idx)->parent;

    while (nodeIdxParent != nullptr) {
        const int old = atomicCAS(&(nodeIdxParent->updateFlag), 0, 1);
        if (old == 0) {
            /* first thread entered here. 
                Wait the other thread from the other child node. */ 
            return;
        }
        assert(old == 1);
        /* old has been one, another thead can access here. merge its child's AABB boxes. */
        nodeIdxParent->bbox = merge(nodeIdxParent->leftChild->bbox, nodeIdxParent->rightChild->bbox);

        #if DEBUG_INFO
        printf("* parent idx (%d) bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n"
                "* leftChild is_leaf(%d) bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n"
                "* rightChild is_leaf(%d) bmin(%0.6f,%0.6f,%0.6f) bmax(%0.6f,%0.6f,%0.6f) \n",
                nodeIdxParent - internalNodes, nodeIdxParent->bbox.bmin.x , nodeIdxParent->bbox.bmin.y,nodeIdxParent->bbox.bmin.z, nodeIdxParent->bbox.bmax.x , nodeIdxParent->bbox.bmax.y, nodeIdxParent->bbox.bmax.z,
                nodeIdxParent->leftChild->isLeaf, nodeIdxParent->leftChild->bbox.bmin.x, nodeIdxParent->leftChild->bbox.bmin.y , nodeIdxParent->leftChild->bbox.bmin.z, nodeIdxParent->leftChild->bbox.bmax.x, nodeIdxParent->leftChild->bbox.bmax.y, nodeIdxParent->leftChild->bbox.bmax.z ,
                nodeIdxParent->rightChild->isLeaf, nodeIdxParent->rightChild->bbox.bmin.x, nodeIdxParent->rightChild->bbox.bmin.y , nodeIdxParent->rightChild->bbox.bmin.z, nodeIdxParent->rightChild->bbox.bmax.x, nodeIdxParent->rightChild->bbox.bmax.y, nodeIdxParent->rightChild->bbox.bmax.z );
        #endif
        
        /* reading global memory is a blocking process, but writing action doesn't. The thread
            will continue working rather than wait until the writing completed. */
        __threadfence();
        /* get next parent */
        nodeIdxParent = nodeIdxParent->parent;
    }

    return;
}


__global__ void 
NbInfoScan_Kernel_First(std::uint32_t num_object, AABB* aabbs, Node* internalNodes, 
        std::uint32_t* adjObjNums, std::uint32_t* adjObjectsOut, std::uint32_t* prefix_sum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_object - 1) {
        return;
    }

    /* query for each object's neighbour */
    std::uint32_t adjObjNum = lbvh::query_device(aabbs + idx, internalNodes, nullptr, true);
    adjObjNums[idx] = adjObjNum;

}

__global__ void 
NbInfoScan_Kernel_Second(std::uint32_t num_object, AABB* aabbs, Node* internalNodes, 
        std::uint32_t* adjObjNums, std::uint32_t* adjObjectsOut, std::uint32_t* prefix_sum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_object - 1) {
        return;
    }

    /* query for each object's neighbour */
    lbvh::query_device(aabbs + idx, internalNodes, 
                        adjObjectsOut + prefix_sum[idx], 
                        false,
                        adjObjNums[idx]);    
}



void cudaDevInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int dev;
    for (dev = 0; dev < deviceCount; dev++)
    {
        int driver_version(0), runtime_version(0);

        size_t available, total;
        cudaMemGetInfo(&available, &total);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
            if (deviceProp.minor = 9999 && deviceProp.major == 9999)
                printf("\n");

        printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driver_version);
        printf("CUDA Driver Version:                            %d.%d\n", 
            driver_version / 1000, (driver_version % 1000) / 10);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA Runtime Version:                           %d.%d\n", 
            runtime_version / 1000, (runtime_version % 1000) / 10);
        printf("Device Prop:                                    %d.%d\n", 
            deviceProp.major, deviceProp.minor);
        printf("Total amount of Global Memory:                  %lu bytes\n", 
            deviceProp.totalGlobalMem);
        printf("Total amount of AVALUABLE Memory:               %lu bytes\n", 
            available);
        printf("Number of SMs:                                  %d\n", 
            deviceProp.multiProcessorCount);
        printf("Total amount of Constant Memory:                %lu bytes\n", 
            deviceProp.totalConstMem);
        printf("Total amount of Shared Memory per block:        %lu bytes\n", 
            deviceProp.sharedMemPerBlock);
        printf("Total number of registers available per block:  %d\n", 
            deviceProp.regsPerBlock);
        printf("Warp size:                                      %d\n", 
            deviceProp.warpSize);
        printf("Maximum number of threads per SM:               %d\n", 
            deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block:            %d\n", 
            deviceProp.maxThreadsPerBlock);
        printf("Maximum size of each dimension of a block:      %d x %d x %d\n", 
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", 
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch:                           %lu bytes\n", 
            deviceProp.memPitch);
        printf("Texture alignmemt:                              %lu bytes\n", 
            deviceProp.texturePitchAlignment);
        printf("Clock rate:                                     %.2f GHz\n", 
            deviceProp.clockRate * 1e-6f);
        printf("Memory Clock rate:                              %.0f MHz\n", 
            deviceProp.memoryClockRate * 1e-3f);
        printf("Memory Bus Width:                               %d-bit\n\n", 
            deviceProp.memoryBusWidth);
    }
}

}





