#pragma once

#ifndef VEC3_H
#define VEC3_H

struct vec3 {
    double x;
    double y;
    double z;

    [[nodiscard]] double length() const noexcept;

    vec3 &operator-=(const vec3 &rhs) noexcept;
};

/*! \brief Subtract two vectors of double
 *  \param[in] lhs Minuend of subtraction
 *  \param[in] rhs Subtrahend of subtraction
 *  \returns Result of subtraction
 */
vec3 operator-(const vec3 &lhs, const vec3 &rhs) noexcept;

/*! \brief Compute dot-product of two vectors
 *  \param[in] lhs First multiplicand
 *  \param[in] rhs Second multiplicand
 *  \returns Dot-product of both multiplicands
 */
double operator*(const vec3 &lhs, const vec3 &rhs) noexcept;

/*! \brief Compute vector multiply by scalar
 *  \param[in] v Vector multiplicand
 *  \param[in] scalar Scalar multiplicand
 *  \returns Result of multiplication
 */
vec3 operator*(const vec3 &v, double scalar) noexcept;

/*! \brief Compute vector multiply by scalar
 *  \param[in] scalar Scalar multiplicand
 *  \param[in] v Vector multiplicand
 *  \returns Result of multiplication
 */
vec3 operator*(double scalar, const vec3 &v) noexcept;

#endif
