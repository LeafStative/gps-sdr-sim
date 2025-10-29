#include "gpstime.h"

#include <cmath>

double gpstime_t::total_seconds() const noexcept {
    return week * SECONDS_IN_WEEK + sec;
}

gpstime_t &gpstime_t::operator-=(const double dt) noexcept {
    return *this += -dt;
}

gpstime_t &gpstime_t::operator+=(const double dt) noexcept {
    auto seconds = sec + dt;
    seconds      = std::round(seconds * 1000.0) / 1000.0; // Avoid rounding error

    const auto weeks = std::floor(seconds / SECONDS_IN_WEEK);

    sec = seconds - weeks * SECONDS_IN_WEEK;
    week += static_cast<int>(weeks);

    return *this;
}

double operator-(const gpstime_t &lhs, const gpstime_t &rhs) noexcept {
    const gpstime_t tmp{
        .week = lhs.week - rhs.week, //
        .sec  = lhs.sec - rhs.sec};

    return tmp.total_seconds();
}

gpstime_t operator+(const gpstime_t &lhs, const double dt) noexcept {
    gpstime_t tmp(lhs);
    return tmp += dt;
}
