#include "tiny_obj_loader.h"

#include "book.h"
#include "bvh.cuh"
#include "morton_code.cuh"
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>


namespace lbvh {

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



/* Kernel declaration */
__global__ void 
computeBBoxes_Kernel(const __uint32_t& num_objects, triangle_t* trianglePtr, vec3f* verticePtr, AABB* aabbPtr);

__global__ void 
computeMortonCode_kernel(__uint32_t num_objects, __uint32_t* objectIDs, 
                            AABB& aabb_bound, AABB* aabbs, __uint32_t* mortonCodes);


__host__ void 
BVH::loadObj(std::string& inputfile) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    /* deal with unexpected situation */
    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    // Loop over shapes
    size_t shapes_size = shapes.size();
    for (size_t s = 0; s < shapes_size; s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
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
    for (size_t s = 0; s < vertices_size; s++) {
        vec3f vertice;
        vertice.x = attrib.vertices.at(s * 3 + 0);
        vertice.y = attrib.vertices.at(s * 3 + 1);
        vertice.z = attrib.vertices.at(s * 3 + 2);
        vertices_h_.push_back(vertice);
    }

    /* get normals */
    size_t normals_size = attrib.normals.size();
    for (size_t s = 0; s < normals_size; s++) {
        vec3f normal;
        normal.x = attrib.normals.at(s * 3 + 0);
        normal.y = attrib.normals.at(s * 3 + 1);
        normal.z = attrib.normals.at(s * 3 + 2);
        normals_h_.push_back(normal);
    }
    
}


__host__ void 
BVH::construct(std::string inputfile) {
    if(triangle_indices_h_.size() == 0u || 
        vertices_h_.size() == 0u || 
        normals_h_.size() == 0u ) {

        printf("Please load objects fisrt. Error happens in %s at line %d.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    /* ---------------- STAGE 1: load objects ---------------- */
    /* allocte specific memory size */
    HANDLE_ERROR(cudaMalloc((void**)&triangle_indices_d_, triangle_indices_h_.size() * sizeof(triangle_t)));
    HANDLE_ERROR(cudaMalloc((void**)&vertices_d_, vertices_h_.size() * sizeof(vec3f)));
    HANDLE_ERROR(cudaMalloc((void**)&normals_d_, vertices_h_.size() * sizeof(vec3f)));
    HANDLE_ERROR(cudaMalloc((void**)&aabbs_d_, triangle_indices_h_.size() * sizeof(AABB)));

    /* copy data from host to device */
    HANDLE_ERROR(cudaMemcpy(&triangle_indices_d_, triangle_indices_h_.data(), 
                            triangle_indices_h_.size() * sizeof(triangle_t), 
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(&vertices_d_, vertices_h_.data(), 
                            vertices_h_.size() * sizeof(vec3f), 
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(&normals_d_, normals_h_.data(), 
                            vertices_h_.size() * sizeof(vec3f), 
                            cudaMemcpyHostToDevice));


    const __uint32_t num_objects        = triangle_indices_h_.size();
    const __uint32_t num_internal_nodes = num_objects - 1;
    const __uint32_t num_nodes          = num_objects * 2 - 1;

    /* kernel property */
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_objects - 1 + threadsPerBlock - 1) / threadsPerBlock;

    /* construct aabb */
    computeBBoxes_Kernel<<<blocksPerGrid, threadsPerBlock>>>(num_objects, triangle_indices_d_, vertices_d_, aabbs_d_);

    /* calculate morton code for all objects */
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

    printf("found AABB bound min(%0.6f, %0.6f , %0.6f)\n" , aabb_bound.bmin.x , aabb_bound.bmin.y , aabb_bound.bmin.z);
    printf("found AABB bound max(%0.6f, %0.6f , %0.6f)\n" , aabb_bound.bmax.x , aabb_bound.bmax.y , aabb_bound.bmax.z);

    /* ---------------- STAGE 2: build BVH Tree ---------------- */
    HANDLE_ERROR(cudaMalloc(&mortonCodes, num_objects * sizeof(__uint32_t)));
    HANDLE_ERROR(cudaMalloc(&objectIDs, num_objects * sizeof(__uint32_t)));
    HANDLE_ERROR(cudaMalloc(&LeafNodes, num_objects * sizeof(LeafNode)));
    HANDLE_ERROR(cudaMalloc(&internalNodes, (num_objects - 1) * sizeof(InternalNode)));

    /* compute morton code */
    computeMortonCode_kernel<<<blocksPerGrid, threadsPerBlock>>>(num_objects, objectIDs, aabb_bound, aabbs_d_, mortonCodes);

    /* sort morton codes */
    thrust::device_ptr<__uint32_t> mortonCodes_d_ptr(mortonCodes);
    thrust::device_ptr<__uint32_t> objectIDs_d_ptr(objectIDs);
    thrust::sort_by_key(mortonCodes_d_ptr, mortonCodes_d_ptr + num_objects, objectIDs_d_ptr);

    /* check morton codes are unique */
    thrust::device_ptr<__uint64_t> mortonCodes64_d_ptr;
    const auto uniqued = thrust::unique_copy(mortonCodes_d_ptr, mortonCodes_d_ptr + num_objects, mortonCodes64_d_ptr);

    /* construct leaf nodes */
    thrust::device_ptr<LeafNode> leafNodes_d_ptr(LeafNodes);
    thrust::transform(LeafNodes, LeafNodes + num_objects, leafNodes_d_ptr,
        [] __device__ (const __uint32_t idx) {
            LeafNode leaf;
            leaf.setObjectID(idx);
            
            return leaf;
        });

    
}

BVH::~BVH() {
    HANDLE_ERROR(cudaFree(triangle_indices_d_));
    HANDLE_ERROR(cudaFree(vertices_d_));
    HANDLE_ERROR(cudaFree(normals_d_));
    HANDLE_ERROR(cudaFree(aabbs_d_));
}


__global__ void 
computeBBoxes_Kernel(const __uint32_t& num_objects, triangle_t* triangles, vec3f* vertices, AABB* aabbs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) {
        return;
    }
    
    triangle_t tri = triangles[idx];
    vec3f vertex_a = vertices[tri.a.vertex_index];
    vec3f vertex_b = vertices[tri.b.vertex_index];
    vec3f vertex_c = vertices[tri.c.vertex_index];

    vec3f tmp_bmax = vmax(vertex_a, vertex_b);
    vec3f tmp_bmin = vmin(vertex_a, vertex_b);
    aabbs[idx].bmax = vmax(vertex_c, tmp_bmax);
    aabbs[idx].bmin = vmin(vertex_c, tmp_bmin);
}



__global__ void 
computeMortonCode_kernel(__uint32_t num_objects, __uint32_t* objectIDs, 
                            AABB& aabb_bound, AABB* aabbs, __uint32_t* mortonCodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_objects) {
        return;
    }

    objectIDs[idx] = idx;
    vec3f centroid = aabbs[idx].getCentroid();
    centroid.x = (centroid.x - aabb_bound.bmin.x) / (aabb_bound.bmax.x - aabb_bound.bmin.x);
    centroid.y = (centroid.y - aabb_bound.bmin.y) / (aabb_bound.bmax.y - aabb_bound.bmin.y);
    centroid.z = (centroid.z - aabb_bound.bmin.z) / (aabb_bound.bmax.z - aabb_bound.bmin.z);
    mortonCodes[idx] = morton3D(centroid);

    return;
}

__global__ void
constructInternalNodes_kernel() {

}



}

