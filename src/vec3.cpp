#include "vec3.h"

#include <cmath>

double vec3::length() const noexcept {
    return std::sqrt(x * x + y * y + z * z);
}

vec3 &vec3::operator-=(const vec3 &rhs) noexcept {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;

    return *this;
}

vec3 operator-(const vec3 &lhs, const vec3 &rhs) noexcept {
    return vec3{
        .x = lhs.x - rhs.x, //
        .y = lhs.y - rhs.y,
        .z = lhs.z - rhs.z};
}

double operator*(const vec3 &lhs, const vec3 &rhs) noexcept {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

vec3 operator*(const vec3 &v, const double scalar) noexcept {
    return vec3{
        .x = v.x * scalar, //
        .y = v.y * scalar,
        .z = v.z * scalar};
}

vec3 operator*(const double scalar, const vec3 &v) noexcept {
    return v * scalar;
}
