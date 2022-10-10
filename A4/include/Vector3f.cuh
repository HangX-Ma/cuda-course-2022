/**
 * @file Vector3f.cuh
 * @author HangX-Ma m-contour@qq.com
 * @brief Vector3f struct type definition
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

class Vector3f {
public:
    /* constructor */
    __host__ __device__
    Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}

    __host__ __device__
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__
    Vector3f(const Vector3f& v) : x(v.x), y(v.y), z(v.z) {}

    __host__ __device__
    explicit Vector3f(float a) : x(a), y(a), z(a) {}
    
    __host__ __device__
    explicit Vector3f(float* a) : x(a[0]), y(a[1]), z(a[2]) {}

    /* variable */
    float x;
    float y;
    float z;

    /* overload method */
    __host__ __device__ Vector3f& 
    operator=(float a){ x = a; y = a; z = a; return *this; }
    
    __host__ __device__ Vector3f& 
    operator+(float a){ x += a; y += a; z += a; return *this; }

    __host__ __device__ Vector3f& 
    operator*(float a){ x *= a; y *= a; z *= a; return *this; }

    __host__ __device__ Vector3f& 
    operator/(float a){ x /= a; y /= a; z /= a; return *this; }

    __host__ __device__ Vector3f 
    operator-() {return Vector3f(-x, -y, -z);}

    __host__ __device__ Vector3f& 
    operator=(const Vector3f &rhs){ x = rhs.x; y = rhs.y; z = rhs.z; return *this; }

    __host__ __device__ Vector3f& 
    operator+(const Vector3f& rhs){ x += rhs.x; y += rhs.y; z += rhs.z; return *this; }

    __host__ __device__ Vector3f& 
    operator-(const Vector3f& rhs){ x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }

    __host__ __device__ Vector3f& 
    operator*(const Vector3f& rhs){ x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this; }

    __host__ __device__ Vector3f& 
    operator/(const Vector3f& rhs){ x /= rhs.x; y /= rhs.y; z /= rhs.z; return *this; }

    /* quick calculation */
    __host__ __device__ Vector3f& 
    operator+=(const Vector3f& rhs){ x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
    
    __host__ __device__ Vector3f& 
    operator-=(const Vector3f& rhs){ x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }

    __host__ __device__ Vector3f& 
    operator*=(const Vector3f& rhs){ x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this; }

    __host__ __device__ Vector3f& 
    operator/=(const Vector3f& rhs){ x /= rhs.x; y /= rhs.y; z /= rhs.z; return *this; }

    /* cross product */
    __host__ __device__ inline Vector3f
    cross (const Vector3f& rhs) const {
        return Vector3f ( y * rhs.z - z * rhs.y,
                          z * rhs.x - x * rhs.z,
                          x * rhs.y - y * rhs.x);
    }
    
    /* dot product */
    __host__ __device__ inline float
    dot (const Vector3f& rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z; 
    }

    /* get vector length */
    __host__ __device__ inline float
    norm2 () const {
        return sqrtf(dot(*this));
    }

    /* get normalized value */
    __host__ __device__ inline float 
    normalize(){
        return 1.0/sqrtf(x * x + y * y + z * z);
    }


};



#endif