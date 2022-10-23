#ifndef __MORTON_CODE_H__
#define __MORTON_CODE_H__

#include <cstdint>
#include "vector3f.h"

namespace lbvh {


__host__ __device__ __inline__ std::uint32_t
expandBits(std::uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

/**
 * @brief Calculates a 30-bit Morton code for the
 * given 3D point located within the unit cube [0,1].
 * 
 * @param[in] x position value mapped on x-axis
 * @param[in] y position value mapped on y-axis
 * @param[in] z position value mapped on z-axis
 * @return morton code 
 */
__host__ __device__ __inline__ std::uint32_t
morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    const std::uint32_t xx = expandBits((std::uint32_t)x);
    const std::uint32_t yy = expandBits((std::uint32_t)y);
    const std::uint32_t zz = expandBits((std::uint32_t)z);

    return xx | (yy << 1) | (zz << 2);
}


/**
 * @brief Calculates a 30-bit Morton code for the
 * given 3D point located within the unit cube [0,1].
 * 
 * @note morton code 1024, 10 bits = 1024
 * @param[in] vec3f space vector
 * @return morton code 
 */
__host__ __device__ __inline__ std::uint32_t
morton3D(vec3f& vec) {
    vec.x = fminf(fmaxf(vec.x * 1024.0f, 0.0f), 1023.0f);
    vec.y = fminf(fmaxf(vec.y * 1024.0f, 0.0f), 1023.0f);
    vec.z = fminf(fmaxf(vec.z * 1024.0f, 0.0f), 1023.0f);
    const std::uint32_t xx = expandBits((std::uint32_t)vec.x);
    const std::uint32_t yy = expandBits((std::uint32_t)vec.y);
    const std::uint32_t zz = expandBits((std::uint32_t)vec.z);

    return xx | (yy << 1) | (zz << 2);
}

}
#endif