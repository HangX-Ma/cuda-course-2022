/**
 * @file vector3f.cuh
 * @author HangX-Ma m-contour@qq.com
 * @brief vec3f struct type definition
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


#ifndef __VECTOR3_CUH__
#define __VECTOR3_CUH__

namespace lbvh {

constexpr float GLH_EPSILON = 1e-6f;

typedef struct vec3f vec3f;

struct vec3f {
    /* constructor */
    __host__ __device__
    vec3f() : x(0.0f), y(0.0f), z(0.0f) {}

    __host__ __device__
    vec3f(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__
    vec3f(const vec3f& v) : x(v.x), y(v.y), z(v.z) {}

    __host__ __device__
    explicit vec3f(float a) : x(a), y(a), z(a) {}
    
    __host__ __device__
    explicit vec3f(float* a) : x(a[0]), y(a[1]), z(a[2]) {}

    /* variable */
    float x, y, z;

    /* overload method */
    __host__ __device__ vec3f
    operator+(float a) const { return vec3f(x + a, y + a, z + a); }

    __host__ __device__ vec3f
    operator*(float a) const { return vec3f(x * a, y * a, z * a); }

    __host__ __device__ vec3f 
    operator/(float a) const { return vec3f(x / a, y / a, z / a); }

    __host__ __device__ vec3f 
    operator-() const {return vec3f(-x, -y, -z);}

    __host__ __device__ vec3f&
    operator=(const vec3f &rhs){ x = rhs.x; y = rhs.y; z = rhs.z; return *this; }

    __host__ __device__ vec3f
    operator+(const vec3f& rhs) const { return vec3f(x + rhs.x, y + rhs.y, z + rhs.z); }

    __host__ __device__ vec3f
    operator-(const vec3f& rhs) const { return vec3f(x - rhs.x, y - rhs.y, z - rhs.z); }

    __host__ __device__ vec3f
    operator*(const vec3f& rhs) const { return vec3f(x * rhs.x, y * rhs.y, z * rhs.z); }

    __host__ __device__ vec3f
    operator/(const vec3f& rhs) const { return vec3f(x / rhs.x, y / rhs.y, z / rhs.z); }

    /* quick calculation */
    __host__ __device__ vec3f& 
    operator+=(const vec3f& rhs){ x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
    
    __host__ __device__ vec3f& 
    operator-=(const vec3f& rhs){ x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }

    __host__ __device__ vec3f& 
    operator*=(const vec3f& rhs){ x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this; }

    __host__ __device__ vec3f& 
    operator/=(const vec3f& rhs){ x /= rhs.x; y /= rhs.y; z /= rhs.z; return *this; }

    __host__ __device__ __inline__ vec3f& 
    set_value( const float &vx, const float &vy, const float &vz) { 
        x = vx; y = vy; z = vz; 
        return *this; 
    }

    __host__ __device__ __inline__ vec3f&
    zero() {
        x = 0.0f; y = 0.0f; z = 0.0f;
        return *this;
    }

    /* cross product */
    __host__ __device__ __inline__ vec3f
    cross (const vec3f& rhs) const {
        return vec3f ( y * rhs.z - z * rhs.y,
                          z * rhs.x - x * rhs.z,
                          x * rhs.y - y * rhs.x);
    }
    
    /* dot product */
    __host__ __device__ __inline__ float
    dot (const vec3f& rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z; 
    }

    /* get vector length */
    __host__ __device__ __inline__ float
    length () const {
        return sqrtf(dot(*this));
    }

    /* get normalized value */
    __host__ __device__ __inline__ void 
    normalize() {
        double sum = x * x + y * y + z * z;
        if (sum > GLH_EPSILON) {
        double base =  1.0/sqrtf(sum);
            x *= base;
            y *= base;
            z *= base;
        }
    }

};


__host__ __device__ __inline__ vec3f
vmin(vec3f& lhs, vec3f& rhs) {
    return vec3f(fminf(lhs.x, rhs.x), fminf(lhs.y, rhs.y), fminf(lhs.z, rhs.z));
}

__host__ __device__ __inline__ vec3f
vmax(vec3f& lhs, vec3f& rhs) {
    return vec3f(fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y), fmaxf(lhs.z, rhs.z));
}

}

#endif