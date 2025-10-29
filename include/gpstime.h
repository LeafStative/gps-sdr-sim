#pragma once

#ifndef GPSTIME_H
#define GPSTIME_H

constexpr auto SECONDS_IN_WEEK      = 604800.0;
constexpr auto SECONDS_IN_HALF_WEEK = 302400.0;
constexpr auto SECONDS_IN_DAY       = 86400.0;
constexpr auto SECONDS_IN_HOUR      = 3600.0;
constexpr auto SECONDS_IN_MINUTE    = 60.0;

/*! \brief Structure representing GPS time */
struct gpstime_t {
    int    week; /*!< GPS week number (since January 1980) */
    double sec;  /*!< second inside the GPS \a week */

    [[nodiscard]] double total_seconds() const noexcept;

    gpstime_t &operator-=(double dt) noexcept;
    gpstime_t &operator+=(double dt) noexcept;
};

double    operator-(const gpstime_t &lhs, const gpstime_t &rhs) noexcept;
gpstime_t operator+(const gpstime_t &lhs, double dt) noexcept;

#endif
