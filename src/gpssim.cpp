#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <charconv>
#include <chrono>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cxxopts.hpp>
#include <scn/scan.h>

#include "gpssim.h"

namespace ranges = std::ranges;
namespace views  = std::views;
namespace chrono = std::chrono;

namespace {

// clang-format off
constexpr auto SIN_TABLE512 = std::to_array({
       2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
      50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
      97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
     140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
     178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
     209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
     232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
     245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
     250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
     245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
     230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
     207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
     176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
     138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
      94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
      47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
      -2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
     -50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
     -97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
    -140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
    -178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
    -209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
    -232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
    -245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
    -250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
    -245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
    -230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
    -207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
    -176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
    -138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
     -94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
     -47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
});

constexpr auto COS_TABLE512 = std::to_array({
     250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
     245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
     230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
     207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
     176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
     138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
      94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
      47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
      -2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
     -50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
     -97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
    -140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
    -178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
    -209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
    -232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
    -245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
    -250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
    -245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
    -230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
    -207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
    -176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
    -138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
     -94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
     -47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
       2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
      50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
      97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
     140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
     178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
     209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
     232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
     245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
});

// Receiver antenna attenuation in dB for boresight angle = 0:5:180 [deg]
constexpr auto ANT_PAT_DB = std::to_array({
     0.00,  0.00,  0.22,  0.44,  0.67,  1.11,  1.56,  2.00,  2.44,  2.89,  3.56,  4.22,
     4.89,  5.56,  6.22,  6.89,  7.56,  8.22,  8.89,  9.78, 10.67, 11.56, 12.44, 13.33,
    14.44, 15.56, 16.67, 17.78, 18.89, 20.00, 21.33, 22.67, 24.00, 25.56, 27.33, 29.33,
    31.56
});
// clang-format on

std::array<int, MAX_SAT>           allocated_sat;
std::array<vec3, USER_MOTION_SIZE> xyz;

/* !\brief generate the C/A code sequence for a given Satellite Vehicle PRN
 *  \param[in] prn PRN number of the Satellite Vehicle
 *  \param[out] ca Caller-allocated integer array of 1023 bytes
 */
void codegen(std::span<int, CA_SEQ_LEN> ca, const int prn) {
    constexpr std::array<size_t, 32> delay{5,   6,   7,   8,   17,  18,  139, 140, 141, 251, 252, 254, 255, 256, 257, 258,
                                           469, 470, 471, 472, 473, 474, 509, 512, 513, 514, 515, 516, 859, 860, 861, 862};

    if (prn < 1 || prn > 32) {
        return;
    }

    int r1[N_DWORD_SBF], r2[N_DWORD_SBF];
    for (size_t i = 0; i < N_DWORD_SBF; ++i) {
        r1[i] = r2[i] = -1;
    }

    int g1[CA_SEQ_LEN], g2[CA_SEQ_LEN];
    for (size_t i = 0; i < CA_SEQ_LEN; ++i) {
        g1[i] = r1[N_DWORD_SBF - 1];
        g2[i] = r2[N_DWORD_SBF - 1];

        const auto c1 = r1[2] * r1[N_DWORD_SBF - 1];
        const auto c2 = r2[1] * r2[2] * r2[5] * r2[7] * r2[8] * r2[N_DWORD_SBF - 1];

        for (int j = N_DWORD_SBF - 1; j > 0; --j) {
            r1[j] = r1[j - 1];
            r2[j] = r2[j - 1];
        }
        r1[0] = c1;
        r2[0] = c2;
    }

    for (size_t i = 0, j = CA_SEQ_LEN - delay[prn - 1]; i < CA_SEQ_LEN; ++i, ++j) {
        ca[i] = (1 - g1[i] * g2[j % CA_SEQ_LEN]) / 2;
    }
}

/*! \brief Convert a UTC date into a GPS date
 *  \param[in] t input date in UTC form
 *  \return g output date in GPS form
 */
gpstime_t date2gps(const datetime_t &t) {
    constexpr auto doy = std::to_array({0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334});

    const auto year = t.y - 1980;

    // Compute the number of leap days since Jan 5/Jan 6, 1980.
    const auto leap_days = year % 4 == 0 && t.m <= 2 ? year / 4 : year / 4 + 1;

    // Compute the number of days elapsed since Jan 5/Jan 6, 1980.
    const auto days_elapsed = year * 365 + doy[t.m - 1] + t.d + leap_days - 6;

    // Convert time to GPS weeks and seconds.
    return gpstime_t{
        .week = days_elapsed / 7,
        .sec =
            static_cast<double>(days_elapsed % 7) * SECONDS_IN_DAY + t.hh * SECONDS_IN_HOUR + t.mm * SECONDS_IN_MINUTE + t.sec};
}

/*! \brief Convert a UTC date into a GPS date
 *  \param[in] g input date in GPS form
 *  \return g output date in UTC form
 */
datetime_t gps2date(const gpstime_t &g) {
    // Convert Julian day number to calendar date
    const auto c = static_cast<int>(7 * g.week + std::floor(g.sec / 86400.0) + 2444245.0) + 1537;
    const auto d = static_cast<int>((c - 122.1) / 365.25);
    const auto e = 365 * d + d / 4;
    const auto f = static_cast<int>((c - e) / 30.6001);

    const auto minutes = f - 1 - 12 * (f / 14);
    return datetime_t{
        .y   = d - 4715 - (7 + minutes) / 10,
        .m   = f - minutes,
        .d   = c - e - static_cast<int>(30.6001 * f),
        .hh  = static_cast<int>(g.sec / 3600.0) % 24,
        .mm  = static_cast<int>(g.sec / 60.0) % 60,
        .sec = g.sec - 60.0 * std::floor(g.sec / 60.0)};
}

/*! \brief Convert Earth-centered Earth-fixed (ECEF) into Lat/Long/Height
 *  \param[in] xyz Input vector of X, Y and Z ECEF coordinates
 *  \return Output vector of Latitude, Longitude and Height
 */
vec3 xyz2llh(const vec3 &xyz) {
    constexpr auto a = WGS84_RADIUS;
    constexpr auto e = WGS84_ECCENTRICITY;

    constexpr auto eps = 1.0e-3;
    constexpr auto e2  = e * e;

    if (xyz.length() < eps) {
        // Invalid ECEF vector
        return vec3{0.0, 0.0, -a};
    }

    const auto &[x, y, z] = xyz;

    const auto rho2 = x * x + y * y;
    auto       dz   = e2 * z;

    double zdz, nh, n;
    while (true) {
        zdz               = z + dz;
        nh                = std::sqrt(rho2 + zdz * zdz);
        const auto slat   = zdz / nh;
        n                 = a / std::sqrt(1.0 - e2 * slat * slat);
        const auto dz_new = n * e2 * slat;

        if (std::abs(dz - dz_new) < eps) {
            break;
        }
        dz = dz_new;
    }

    return vec3{
        std::atan2(zdz, std::sqrt(rho2)), //
        std::atan2(y, x),
        nh - n};
}

/*! \brief Convert Lat/Long/Height into Earth-centered Earth-fixed (ECEF)
 *  \param[in] llh Input vector of Latitude, Longitude and Height
 *  \return Output vector of X, Y and Z ECEF coordinates
 */
vec3 llh2xyz(const vec3 &llh) {
    constexpr auto a  = WGS84_RADIUS;
    constexpr auto e  = WGS84_ECCENTRICITY;
    constexpr auto e2 = e * e;

    const auto c_lat = std::cos(llh.x);
    const auto s_lat = std::sin(llh.x);
    const auto c_lon = std::cos(llh.y);
    const auto s_lon = std::sin(llh.y);
    const auto d     = e * s_lat;

    const auto n   = a / std::sqrt(1.0 - d * d);
    const auto nph = n + llh.z;

    const auto tmp = nph * c_lat;

    return vec3{
        tmp * c_lon, //
        tmp * s_lon,
        ((1.0 - e2) * n + llh.z) * s_lat};
}

/*! \brief Compute the intermediate matrix for LLH to ECEF
 *  \param[in] llh Input position in Latitude-Longitude-Height format
 *  \param[out] t Three-by-Three output matrix
 */
void ltcmat(const vec3 &llh, double t[3][3]) {
    const double s_lat = std::sin(llh.x);
    const double c_lat = std::cos(llh.x);
    const double s_lon = std::sin(llh.y);
    const double c_lon = std::cos(llh.y);

    t[0][0] = -s_lat * c_lon;
    t[0][1] = -s_lat * s_lon;
    t[0][2] = c_lat;
    t[1][0] = -s_lon;
    t[1][1] = c_lon;
    t[1][2] = 0.0;
    t[2][0] = c_lat * c_lon;
    t[2][1] = c_lat * s_lon;
    t[2][2] = s_lat;
}

/*! \brief Convert Earth-centered Earth-Fixed to ?
 *  \param[in] ecef Input position as vector in ECEF format
 *  \param[in] t Intermediate matrix computed by \ref ltcmat
 *  \return Output position as North-East-Up format
 */
vec3 ecef2neu(const vec3 &ecef, double t[3][3]) {
    return vec3{
        t[0][0] * ecef.x + t[0][1] * ecef.y + t[0][2] * ecef.z,
        t[1][0] * ecef.x + t[1][1] * ecef.y + t[1][2] * ecef.z,
        t[2][0] * ecef.x + t[2][1] * ecef.y + t[2][2] * ecef.z};
}

/*! \brief Convert North-East-Up to Azimuth + Elevation
 *  \param[in] neu Input position in North-East-Up format
 *  \param[out] azel Output array of azimuth + elevation as double
 */
void neu2azel(const vec3 &neu, const std::span<double, 2> azel) {
    auto azimuth = std::atan2(neu.y, neu.x);
    if (azimuth < 0.0) {
        azimuth += 2.0 * PI;
    }

    const auto ne = std::sqrt(neu.x * neu.x + neu.y * neu.y);
    azel[0]       = azimuth;
    azel[1]       = std::atan2(neu.z, ne);
}

/*! \brief Compute Satellite position, velocity and clock at given time
 *  \param[in] eph Ephemeris data of the satellite
 *  \param[in] g GPS time at which position is to be computed
 *  \param[out] pos Computed position (vector)
 *  \param[out] vel Computed velocity (vector)
 *  \param[out] clk Computed clock
 */
void satpos(const ephem_t &eph, const gpstime_t &g, vec3 &pos, vec3 &vel, std::span<double, 2> clk) {
    // Computing Satellite Velocity using the Broadcast Ephemeris
    // http://www.ngs.noaa.gov/gps-toolbox/bc_velo.htm

    auto tk = g.sec - eph.toe.sec;
    if (tk > SECONDS_IN_HALF_WEEK) {
        tk -= SECONDS_IN_WEEK;
    } else if (tk < -SECONDS_IN_HALF_WEEK) {
        tk += SECONDS_IN_WEEK;
    }

    const auto mk     = eph.m0 + eph.n * tk;
    auto       ek     = mk;
    auto       ek_old = ek + 1.0;

    auto one_minus_ecos_e = 0.0; // Suppress the uninitialized warning.
    while (std::abs(ek - ek_old) > 1.0E-14) {
        ek_old           = ek;
        one_minus_ecos_e = 1.0 - eph.ecc * std::cos(ek_old);
        ek               = ek + (mk - ek_old + eph.ecc * std::sin(ek_old)) / one_minus_ecos_e;
    }

    const auto sek = std::sin(ek);
    const auto cek = std::cos(ek);

    const auto ek_dot = eph.n / one_minus_ecos_e;

    const auto relativistic = -4.442807633E-10 * eph.ecc * eph.sqrta * sek;

    const auto pk     = std::atan2(eph.sq1e2 * sek, cek - eph.ecc) + eph.aop;
    const auto pk_dot = eph.sq1e2 * ek_dot / one_minus_ecos_e;

    const auto s_2pk = std::sin(2.0 * pk);
    const auto c_2pk = std::cos(2.0 * pk);

    const auto uk     = pk + eph.cus * s_2pk + eph.cuc * c_2pk;
    const auto s_uk   = std::sin(uk);
    const auto c_uk   = std::cos(uk);
    const auto uk_dot = pk_dot * (1.0 + 2.0 * (eph.cus * c_2pk - eph.cuc * s_2pk));

    const auto rk     = eph.A * one_minus_ecos_e + eph.crc * c_2pk + eph.crs * s_2pk;
    const auto rk_dot = eph.A * eph.ecc * sek * ek_dot + 2.0 * pk_dot * (eph.crs * c_2pk - eph.crc * s_2pk);

    const auto ik     = eph.inc0 + eph.idot * tk + eph.cic * c_2pk + eph.cis * s_2pk;
    const auto s_ik   = std::sin(ik);
    const auto c_ik   = std::cos(ik);
    const auto ik_dot = eph.idot + 2.0 * pk_dot * (eph.cis * c_2pk - eph.cic * s_2pk);

    const auto xpk     = rk * c_uk;
    const auto ypk     = rk * s_uk;
    const auto xpk_dot = rk_dot * c_uk - ypk * uk_dot;
    const auto ypk_dot = rk_dot * s_uk + xpk * uk_dot;

    const auto ok   = eph.omg0 + tk * eph.omgkdot - OMEGA_EARTH * eph.toe.sec;
    const auto s_ok = std::sin(ok);
    const auto c_ok = std::cos(ok);

    pos = vec3{
        xpk * c_ok - ypk * c_ik * s_ok, //
        xpk * s_ok + ypk * c_ik * c_ok,
        ypk * s_ik};

    const auto tmp = ypk_dot * c_ik - ypk * s_ik * ik_dot;

    vel = vec3{
        -eph.omgkdot * pos.y + xpk_dot * c_ok - tmp * s_ok,
        eph.omgkdot * pos.x + xpk_dot * s_ok + tmp * c_ok,
        ypk * c_ik * ik_dot + ypk_dot * s_ik};

    // Satellite clock correction
    tk = g.sec - eph.toc.sec;

    if (tk > SECONDS_IN_HALF_WEEK) {
        tk -= SECONDS_IN_WEEK;
    } else if (tk < -SECONDS_IN_HALF_WEEK) {
        tk += SECONDS_IN_WEEK;
    }

    clk[0] = eph.af0 + tk * (eph.af1 + tk * eph.af2) + relativistic - eph.tgd;
    clk[1] = eph.af1 + 2.0 * tk * eph.af2;
}

/*! \brief Compute Subframe from Ephemeris
 *  \param[in] eph Ephemeris of given SV
 *  \param[out] sbf Array of five sub-frames, 10 long words each
 */
void eph2sbf(const ephem_t &eph, const ionoutc_t &ionoutc, unsigned long sbf[5][N_DWORD_SBF]) {
    // FIXED: This has to be the "transmission" week number, not for the ephemeris reference time
    // wn = (unsigned long)(eph.toe.week%1024);
    constexpr auto wn  = 0UL;
    constexpr auto ura = 0UL;

    const auto code_l2 = static_cast<unsigned long>(eph.codeL2);
    const auto svhlth  = static_cast<unsigned long>(eph.svhlth);
    const auto iodc    = static_cast<unsigned long>(eph.iodc);
    const auto tgd     = static_cast<long>(eph.tgd / POW2_M31);
    const auto toc     = static_cast<unsigned long>(eph.toc.sec / 16.0);
    const auto af0     = static_cast<long>(eph.af0 / POW2_M31);
    const auto af1     = static_cast<long>(eph.af1 / POW2_M43);
    const auto af2     = static_cast<long>(eph.af2 / POW2_M55);

    // Subframe 1
    sbf[0][0] = 0x8B0000UL << 6;
    sbf[0][1] = 0x1UL << 8;
    sbf[0][2] =
        (wn & 0x3FFUL) << 20 | (code_l2 & 0x3UL) << 18 | (ura & 0xFUL) << 14 | (svhlth & 0x3FUL) << 8 | (iodc >> 8 & 0x3UL) << 6;
    sbf[0][3] = 0UL;
    sbf[0][4] = 0UL;
    sbf[0][5] = 0UL;
    sbf[0][6] = (tgd & 0xFFUL) << 6;
    sbf[0][7] = (iodc & 0xFFUL) << 22 | (toc & 0xFFFFUL) << 6;
    sbf[0][8] = (af2 & 0xFFUL) << 22 | (af1 & 0xFFFFUL) << 6;
    sbf[0][9] = (af0 & 0x3FFFFFUL) << 8;

    const auto iode   = static_cast<unsigned long>(eph.iode);
    const auto crs    = static_cast<long>(eph.crs / POW2_M5);
    const auto deltan = static_cast<long>(eph.deltan / POW2_M43 / PI);
    const auto m0     = static_cast<long>(eph.m0 / POW2_M31 / PI);
    const auto cuc    = static_cast<long>(eph.cuc / POW2_M29);
    const auto ecc    = static_cast<unsigned long>(eph.ecc / POW2_M33);
    const auto cus    = static_cast<long>(eph.cus / POW2_M29);
    const auto sqrta  = static_cast<unsigned long>(eph.sqrta / POW2_M19);
    const auto toe    = static_cast<unsigned long>(eph.toe.sec / 16.0);

    // Subframe 2
    sbf[1][0] = 0x8B0000UL << 6;
    sbf[1][1] = 0x2UL << 8;
    sbf[1][2] = (iode & 0xFFUL) << 22 | (crs & 0xFFFFUL) << 6;
    sbf[1][3] = (deltan & 0xFFFFUL) << 14 | (m0 >> 24 & 0xFFUL) << 6;
    sbf[1][4] = (m0 & 0xFFFFFFUL) << 6;
    sbf[1][5] = (cuc & 0xFFFFUL) << 14 | (ecc >> 24 & 0xFFUL) << 6;
    sbf[1][6] = (ecc & 0xFFFFFFUL) << 6;
    sbf[1][7] = (cus & 0xFFFFUL) << 14 | (sqrta >> 24 & 0xFFUL) << 6;
    sbf[1][8] = (sqrta & 0xFFFFFFUL) << 6;
    sbf[1][9] = (toe & 0xFFFFUL) << 14;

    const auto cic    = static_cast<long>(eph.cic / POW2_M29);
    const auto cis    = static_cast<long>(eph.cis / POW2_M29);
    const auto omg0   = static_cast<long>(eph.omg0 / POW2_M31 / PI);
    const auto inc0   = static_cast<long>(eph.inc0 / POW2_M31 / PI);
    const auto crc    = static_cast<long>(eph.crc / POW2_M5);
    const auto aop    = static_cast<long>(eph.aop / POW2_M31 / PI);
    const auto omgdot = static_cast<long>(eph.omgdot / POW2_M43 / PI);
    const auto idot   = static_cast<long>(eph.idot / POW2_M43 / PI);

    // Subframe 3
    sbf[2][0] = 0x8B0000UL << 6;
    sbf[2][1] = 0x3UL << 8;
    sbf[2][2] = (cic & 0xFFFFUL) << 14 | (omg0 >> 24 & 0xFFUL) << 6;
    sbf[2][3] = (omg0 & 0xFFFFFFUL) << 6;
    sbf[2][4] = (cis & 0xFFFFUL) << 14 | (inc0 >> 24 & 0xFFUL) << 6;
    sbf[2][5] = (inc0 & 0xFFFFFFUL) << 6;
    sbf[2][6] = (crc & 0xFFFFUL) << 14 | (aop >> 24 & 0xFFUL) << 6;
    sbf[2][7] = (aop & 0xFFFFFFUL) << 6;
    sbf[2][8] = (omgdot & 0xFFFFFFUL) << 6;
    sbf[2][9] = (iode & 0xFFUL) << 22 | (idot & 0x3FFFUL) << 8;

    constexpr unsigned long data_id = 1UL;
    if (ionoutc.valid) {
        constexpr unsigned long sbf4_page18_sv_id = 56UL;

        const auto alpha0 = static_cast<long>(std::round(ionoutc.alpha0 / POW2_M30));
        const auto alpha1 = static_cast<long>(std::round(ionoutc.alpha1 / POW2_M27));
        const auto alpha2 = static_cast<long>(std::round(ionoutc.alpha2 / POW2_M24));
        const auto alpha3 = static_cast<long>(std::round(ionoutc.alpha3 / POW2_M24));
        const auto beta0  = static_cast<long>(std::round(ionoutc.beta0 / 2048.0));
        const auto beta1  = static_cast<long>(std::round(ionoutc.beta1 / 16384.0));
        const auto beta2  = static_cast<long>(std::round(ionoutc.beta2 / 65536.0));
        const auto beta3  = static_cast<long>(std::round(ionoutc.beta3 / 65536.0));
        const auto a0     = static_cast<long>(std::round(ionoutc.a0 / POW2_M30));
        const auto a1     = static_cast<long>(std::round(ionoutc.a1 / POW2_M50));
        const auto dtls   = ionoutc.dtls;
        const auto tot    = static_cast<unsigned long>(ionoutc.tot / 4096);
        const auto wnt    = static_cast<unsigned long>(ionoutc.wnt % 256);

        // 2016/12/31 (Sat) -> WNlsf = 1929, DN = 7 (http://navigationservices.agi.com/GNSSWeb/)
        // Days are counted from 1 to 7 (Sunday is 1).
        unsigned long wnlsf, dtlsf, dn;
        if (ionoutc.leapen) {
            wnlsf = static_cast<unsigned long>(ionoutc.wnlsf % 256);
            dn    = static_cast<unsigned long>(ionoutc.dn);
            dtlsf = static_cast<unsigned long>(ionoutc.dtlsf);
        } else {
            wnlsf = 1929 % 256;
            dn    = 7;
            dtlsf = 18;
        }

        // Subframe 4, page 18
        sbf[3][0] = 0x8B0000UL << 6;
        sbf[3][1] = 0x4UL << 8;
        sbf[3][2] = data_id << 28 | sbf4_page18_sv_id << 22 | (alpha0 & 0xFFUL) << 14 | (alpha1 & 0xFFUL) << 6;
        sbf[3][3] = (alpha2 & 0xFFUL) << 22 | (alpha3 & 0xFFUL) << 14 | (beta0 & 0xFFUL) << 6;
        sbf[3][4] = (beta1 & 0xFFUL) << 22 | (beta2 & 0xFFUL) << 14 | (beta3 & 0xFFUL) << 6;
        sbf[3][5] = (a1 & 0xFFFFFFUL) << 6;
        sbf[3][6] = (a0 >> 8 & 0xFFFFFFUL) << 6;
        sbf[3][7] = (a0 & 0xFFUL) << 22 | (tot & 0xFFUL) << 14 | (wnt & 0xFFUL) << 6;
        sbf[3][8] = (dtls & 0xFFUL) << 22 | (wnlsf & 0xFFUL) << 14 | (dn & 0xFFUL) << 6;
        sbf[3][9] = (dtlsf & 0xFFUL) << 22;

    } else {
        constexpr auto sbf4_page25_sv_id = 63UL;

        // Subframe 4, page 25
        sbf[3][0] = 0x8B0000UL << 6;
        sbf[3][1] = 0x4UL << 8;
        sbf[3][2] = data_id << 28 | sbf4_page25_sv_id << 22;
        sbf[3][3] = 0UL;
        sbf[3][4] = 0UL;
        sbf[3][5] = 0UL;
        sbf[3][6] = 0UL;
        sbf[3][7] = 0UL;
        sbf[3][8] = 0UL;
        sbf[3][9] = 0UL;
    }

    constexpr auto sbf5_page25_sv_id = 51UL;

    const auto wna = static_cast<unsigned long>(eph.toe.week % 256);
    const auto toa = static_cast<unsigned long>(eph.toe.sec / 4096.0);

    // Subframe 5, page 25
    sbf[4][0] = 0x8B0000UL << 6;
    sbf[4][1] = 0x5UL << 8;
    sbf[4][2] = data_id << 28 | sbf5_page25_sv_id << 22 | (toa & 0xFFUL) << 14 | (wna & 0xFFUL) << 6;
    sbf[4][3] = 0UL;
    sbf[4][4] = 0UL;
    sbf[4][5] = 0UL;
    sbf[4][6] = 0UL;
    sbf[4][7] = 0UL;
    sbf[4][8] = 0UL;
    sbf[4][9] = 0UL;
}

/*! \brief Compute the Checksum for one given word of a subframe
 *  \param[in] source The input data
 *  \param[in] nib Does this word contain non-information-bearing bits?
 *  \returns Computed Checksum
 */
unsigned long compute_checksum(const unsigned long source, const bool nib) {
    /*
    Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
    Bits 29 to  6 = Source data bits, d1, d2, ..., d24
    Bits  5 to  0 = Empty parity bits
    */

    /*
    Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
    Bits 29 to  6 = Data bits transmitted by the SV, D1, D2, ..., D24
    Bits  5 to  0 = Computed parity bits, D25, D26, ..., D30
    */

    /*
                      1            2           3
    bit    12 3456 7890 1234 5678 9012 3456 7890
    ---    -------------------------------------
    D25    11 1011 0001 1111 0011 0100 1000 0000
    D26    01 1101 1000 1111 1001 1010 0100 0000
    D27    10 1110 1100 0111 1100 1101 0000 0000
    D28    01 0111 0110 0011 1110 0110 1000 0000
    D29    10 1011 1011 0001 1111 0011 0100 0000
    D30    00 1011 0111 1010 1000 1001 1100 0000
    */

    constexpr auto b_mask = std::to_array({0x3B1F3480UL, 0x1D8F9A40UL, 0x2EC7CD00UL, 0x1763E680UL, 0x2BB1F340UL, 0x0B7A89C0UL});

    auto       d   = source & 0x3FFFFFC0UL;
    const auto d29 = source >> 31 & 0x1UL;
    const auto d30 = source >> 30 & 0x1UL;

    if (nib) { // Non-information bearing bits for word 2 and 10
        /*
        Solve bits 23 and 24 to preserve parity check
        with zeros in bits 29 and 30.
        */

        if ((d30 + std::popcount(b_mask[4] & d)) % 2) {
            d ^= 0x1UL << 6;
        }
        if ((d29 + std::popcount(b_mask[5] & d)) % 2) {
            d ^= 0x1UL << 7;
        }
    }

    auto result = d;
    if (d30) {
        result ^= 0x3FFFFFC0UL;
    }

    result |= ((d29 + std::popcount(b_mask[0] & d)) % 2) << 5;
    result |= ((d30 + std::popcount(b_mask[1] & d)) % 2) << 4;
    result |= ((d29 + std::popcount(b_mask[2] & d)) % 2) << 3;
    result |= ((d30 + std::popcount(b_mask[3] & d)) % 2) << 2;
    result |= ((d30 + std::popcount(b_mask[4] & d)) % 2) << 1;
    result |= (d29 + std::popcount(b_mask[5] & d)) % 2;

    result &= 0x3FFFFFFFUL;
    // D |= (source & 0xC0000000UL); // Add D29* and D30* from source data bits

    return result;
}

template <typename ParseT = void, typename VarT>
requires(std::is_arithmetic_v<ParseT> || std::is_same_v<ParseT, void>) && std::is_arithmetic_v<VarT>
std::from_chars_result from_chars_rinex(std::string_view str, VarT &value) {
    constexpr auto predicate = [](const auto c) { return !std::isspace(c); };

    const auto start = ranges::find_if(str, predicate);
    str.remove_prefix(std::distance(str.begin(), start));

    const auto end = ranges::find_if(views::reverse(str), predicate);
    str.remove_suffix(std::distance(end.base(), str.end()));

    std::string tmp;
    tmp.reserve(str.size());
    ranges::replace_copy(str, std::back_inserter(tmp), 'D', 'E');

    if constexpr (std::is_same_v<ParseT, void> || std::is_same_v<ParseT, VarT>) {
        return std::from_chars(tmp.c_str(), tmp.c_str() + tmp.size(), value);
    } else {
        ParseT     tmp_value;
        const auto result = std::from_chars(tmp.c_str(), tmp.c_str() + tmp.size(), tmp_value);
        value             = static_cast<VarT>(tmp_value);

        return result;
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
auto from_chars(const std::string_view str, T &value) {
    return std::from_chars(str.data(), str.data() + str.size(), value);
}

/*! \brief Read Ephemeris data from the RINEX Navigation file */
/*  \param[out] eph Array of Output SV ephemeris data
 *  \param[in] filename File name of the RINEX file
 *  \returns Number of sets of ephemerides in the file
 */
int read_rinex_nav_all(ephem_t eph[][MAX_SAT], ionoutc_t &ionoutc, const std::string &filename) {
    std::ifstream fs{filename};
    if (!fs.is_open()) {
        return -1;
    }

    // Clear valid flag
    for (size_t ieph = 0; ieph < EPHEM_ARRAY_SIZE; ++ieph) {
        for (size_t sv = 0; sv < MAX_SAT; ++sv) {
            eph[ieph][sv].valid = false;
        }
    }

    // Read header lines
    std::bitset<4> flags{0b0000};
    while (true) {
        std::string line;
        if (!std::getline(fs, line)) {
            break;
        }
        const std::string_view str = line;

        if (str.substr(60, 13) == "END OF HEADER") {
            break;
        }

        if (str.substr(60, 9) == "ION ALPHA") {
            from_chars_rinex(str.substr(2, 12), ionoutc.alpha0);
            from_chars_rinex(str.substr(14, 12), ionoutc.alpha1);
            from_chars_rinex(str.substr(26, 12), ionoutc.alpha2);
            from_chars_rinex(str.substr(38, 12), ionoutc.alpha3);

            // read wntlsf, dn, and dtlsf from file

            flags.set(0);
        } else if (str.substr(60, 8) == "ION BETA") {
            from_chars_rinex(str.substr(2, 12), ionoutc.beta0);
            from_chars_rinex(str.substr(14, 12), ionoutc.beta1);
            from_chars_rinex(str.substr(26, 12), ionoutc.beta2);
            from_chars_rinex(str.substr(38, 12), ionoutc.beta3);

            flags.set(1);
        } else if (str.substr(60, 9) == "DELTA-UTC") {
            from_chars_rinex(str.substr(3, 19), ionoutc.a0);
            from_chars_rinex(str.substr(22, 19), ionoutc.a1);
            from_chars_rinex(str.substr(41, 9), ionoutc.tot);
            from_chars_rinex(str.substr(50, 9), ionoutc.wnt);

            if (ionoutc.tot % 4096 == 0) {
                flags.set(2);
            }
        } else if (str.substr(60, 12) == "LEAP SECONDS") {
            from_chars_rinex(str.substr(0, 6), ionoutc.dtls);

            flags.set(3);
        }
    }

    // true if read all Iono/UTC lines
    ionoutc.valid = flags.all();

    // Read ephemeris blocks
    gpstime_t g0{.week = -1, .sec = 0};
    size_t    ieph = 0;
    while (true) {
        std::string line;
        if (!std::getline(fs, line)) {
            break;
        }
        std::string_view str = line;

        // PRN
        size_t sv;
        from_chars_rinex(str.substr(0, 2), sv);
        sv -= 1;

        // EPOCH
        datetime_t t;
        from_chars_rinex(str.substr(3, 2), t.y);
        from_chars_rinex(str.substr(6, 2), t.m);
        from_chars_rinex(str.substr(9, 2), t.d);
        from_chars_rinex(str.substr(12, 2), t.hh);
        from_chars_rinex(str.substr(15, 2), t.mm);
        from_chars_rinex(str.substr(18, 4), t.sec);

        t.y += 2000;

        const auto g = date2gps(t);
        if (g0.week == -1) {
            g0 = g;
        }

        // Check current time of clock
        if (const double dt = g - g0; dt > SECONDS_IN_HOUR) {
            g0 = g;
            ++ieph; // a new set of ephemerides

            if (ieph >= EPHEM_ARRAY_SIZE) {
                break;
            }
        }

        auto &ephem = eph[ieph][sv];
        // Date and time
        ephem.t = t;

        // SV CLK
        ephem.toc = g;

        from_chars_rinex(str.substr(22, 19), ephem.af0);
        from_chars_rinex(str.substr(41, 19), ephem.af1);
        from_chars_rinex(str.substr(60, 19), ephem.af2);

        // BROADCAST ORBIT - 1
        if (!std::getline(fs, line)) {
            break;
        }
        str = line;

        from_chars_rinex<double>(str.substr(3, 19), ephem.iode);
        from_chars_rinex(str.substr(22, 19), ephem.crs);
        from_chars_rinex(str.substr(41, 19), ephem.deltan);
        from_chars_rinex(str.substr(60, 19), ephem.m0);

        // BROADCAST ORBIT - 2
        if (!std::getline(fs, line)) {
            break;
        }
        str = line;

        from_chars_rinex(str.substr(3, 19), ephem.cuc);
        from_chars_rinex(str.substr(22, 19), ephem.ecc);
        from_chars_rinex(str.substr(41, 19), ephem.cus);
        from_chars_rinex(str.substr(60, 19), ephem.sqrta);

        // BROADCAST ORBIT - 3
        if (!std::getline(fs, line)) {
            break;
        }
        str = line;

        from_chars_rinex(str.substr(3, 19), ephem.toe.sec);
        from_chars_rinex(str.substr(22, 19), ephem.cic);
        from_chars_rinex(str.substr(41, 19), ephem.omg0);
        from_chars_rinex(str.substr(60, 19), ephem.cis);

        // BROADCAST ORBIT - 4
        if (!std::getline(fs, line)) {
            break;
        }
        str = line;

        from_chars_rinex(str.substr(3, 19), ephem.inc0);
        from_chars_rinex(str.substr(22, 19), ephem.crc);
        from_chars_rinex(str.substr(41, 19), ephem.aop);
        from_chars_rinex(str.substr(60, 19), ephem.omgdot);

        // BROADCAST ORBIT - 5
        if (!std::getline(fs, line)) {
            break;
        }
        str = line;

        from_chars_rinex(str.substr(3, 19), ephem.idot);
        from_chars_rinex<double>(str.substr(22, 19), ephem.codeL2);
        from_chars_rinex<double>(str.substr(41, 19), ephem.toe.week);

        // BROADCAST ORBIT - 6
        if (!std::getline(fs, line)) {
            break;
        }
        str = line;

        from_chars_rinex<double>(str.substr(22, 19), ephem.svhlth);
        from_chars_rinex(str.substr(41, 19), ephem.tgd);
        from_chars_rinex<double>(str.substr(60, 19), ephem.iodc);

        if (ephem.svhlth > 0 && ephem.svhlth < 32) { // Set MSB to 1
            ephem.svhlth += 32;
        }

        // BROADCAST ORBIT - 7
        if (!std::getline(fs, line)) {
            break;
        }

        // Set valid flag
        ephem.valid = true;

        // Update the working variables
        ephem.A       = ephem.sqrta * ephem.sqrta;
        ephem.n       = sqrt(GM_EARTH / (ephem.A * ephem.A * ephem.A)) + ephem.deltan;
        ephem.sq1e2   = sqrt(1.0 - ephem.ecc * ephem.ecc);
        ephem.omgkdot = ephem.omgdot - OMEGA_EARTH;
    }

    if (g0.week >= 0) { // Number of sets of ephemerides
        ++ieph;
    }

    return ieph;
}

double ionospheric_delay(const ionoutc_t &ionoutc, const gpstime_t &g, const vec3 &llh, const std::span<const double, 2> azel) {

    if (!ionoutc.enable) { // No ionospheric delay
        return 0.0;
    }

    const auto e = azel[1] / PI;
    // Obliquity factor
    const auto f = 1.0 + 16.0 * pow(0.53 - e, 3.0);

    if (!ionoutc.valid) {
        return f * 5.0e-9 * SPEED_OF_LIGHT;
    }

    // Earth's central angle between the user position and the earth projection of
    // ionospheric intersection point (semi-circles)
    const auto psi = 0.0137 / (e + 0.11) - 0.022;

    // Geodetic latitude of the earth projection of the ionospheric intersection point
    // (semi-circles)
    const auto phi_u = llh.x / PI;
    const auto phi_i = std::clamp(phi_u + psi * cos(azel[0]), -0.416, 0.416);

    // Geodetic longitude of the earth projection of the ionospheric intersection point
    // (semi-circles)
    const auto lam_u = llh.y / PI;
    const auto lam_i = lam_u + psi * sin(azel[0]) / cos(phi_i * PI);

    // Geomagnetic latitude of the earth projection of the ionospheric intersection
    // point (mean ionospheric height assumed 350 km) (semi-circles)
    const auto phi_m  = phi_i + 0.064 * cos((lam_i - 1.617) * PI);
    const auto phi_m2 = phi_m * phi_m;
    const auto phi_m3 = phi_m2 * phi_m;

    const auto amp = std::max(ionoutc.alpha0 + ionoutc.alpha1 * phi_m + ionoutc.alpha2 * phi_m2 + ionoutc.alpha3 * phi_m3, 0.0);
    const auto per = std::max(ionoutc.beta0 + ionoutc.beta1 * phi_m + ionoutc.beta2 * phi_m2 + ionoutc.beta3 * phi_m3, 72000.0);

    // Local time (sec)
    auto t    = SECONDS_IN_DAY / 2.0 * lam_i + g.sec;
    auto days = std::floor(t / SECONDS_IN_DAY);
    if (t < 0) {
        days = -days - 1;
    }
    t -= days * SECONDS_IN_DAY;

    // Phase (radians)
    if (const auto x = 2.0 * PI * (t - 50400.0) / per; std::abs(x) < 1.57) {
        const auto x2 = x * x;
        const auto x4 = x2 * x2;
        return f * (5.0e-9 + amp * (1.0 - x2 / 2.0 + x4 / 24.0)) * SPEED_OF_LIGHT;
    }

    return f * 5.0e-9 * SPEED_OF_LIGHT;
}

/*! \brief Compute range between a satellite and the receiver
 *  \param[out] rho The computed range
 *  \param[in] eph Ephemeris data of the satellite
 *  \param[in] g GPS time at time of receiving the signal
 *  \param[in] xyz position of the receiver
 */
void compute_range(range_t &rho, const ephem_t &eph, const ionoutc_t &ionoutc, const gpstime_t &g, const vec3 &xyz) {
    // SV position at time of the pseudorange observation.
    vec3                  pos, vel;
    std::array<double, 2> clk;
    satpos(eph, g, pos, vel, clk);

    // Receiver to satellite vector and light-time.
    auto       los = pos - xyz;
    const auto tau = los.length() / SPEED_OF_LIGHT;

    // Extrapolate the satellite position backwards to the transmission time.
    pos -= vel * tau;

    // Earth rotation correction. The change in velocity can be neglected.
    const auto x_rot = pos.x + pos.y * OMEGA_EARTH * tau;
    const auto y_rot = pos.y - pos.x * OMEGA_EARTH * tau;
    pos.x            = x_rot;
    pos.y            = y_rot;

    // New observer to satellite vector and satellite range.
    los              = pos - xyz;
    const auto range = los.length();
    rho.d            = range;

    // Pseudorange.
    rho.range = range - SPEED_OF_LIGHT * clk[0];

    // Relative velocity of SV and receiver.
    const auto rate = vel * los / range;

    // Pseudorange rate.
    rho.rate = rate; // - SPEED_OF_LIGHT * clk[1];

    // Time of application.
    rho.g = g;

    // Azimuth and elevation angles.
    const auto llh = xyz2llh(xyz);

    double tmat[3][3];
    ltcmat(llh, tmat);

    const auto neu = ecef2neu(los, tmat);
    neu2azel(neu, rho.azel);

    // Add ionospheric delay
    rho.iono_delay = ionospheric_delay(ionoutc, g, llh, rho.azel);
    rho.range += rho.iono_delay;
}

/*! \brief Compute the code phase for a given channel (satellite)
 *  \param chan Channel on which we operate (is updated)
 *  \param[in] rho Current range, after \a dt has expired
 *  \param[in dt delta-t (time difference) in seconds
 */
void compute_code_phase(channel_t &chan, const range_t &rho, const double dt) {
    // Pseudorange rate.
    const auto rhorate = (rho.range - chan.rho0.range) / dt;

    // Carrier and code frequency.
    chan.f_carr = -rhorate / LAMBDA_L1;
    chan.f_code = CODE_FREQ + chan.f_carr * CARR_TO_CODE;

    // Initial code phase and data bit counters.
    const auto ms = (chan.rho0.g - chan.g0 + 6.0 - chan.rho0.range / SPEED_OF_LIGHT) * 1000.0;

    auto ims        = static_cast<int>(ms);
    chan.code_phase = (ms - static_cast<double>(ims)) * CA_SEQ_LEN; // in chip

    chan.iword = ims / 600; // 1 word = 30 bits = 600 ms
    ims -= chan.iword * 600;

    chan.ibit = ims / 20; // 1 bit = 20 code = 20 ms
    ims -= chan.ibit * 20;

    chan.icode = ims; // 1 code = 1 ms

    chan.codeCA  = chan.ca[static_cast<int>(chan.code_phase)] * 2 - 1;
    chan.dataBit = static_cast<int>(chan.dwrd[chan.iword] >> (29 - chan.ibit) & 0x1UL) * 2 - 1;

    // Save current pseudorange
    chan.rho0 = rho;
}

/*! \brief Read the list of user motions from the input file
 *  \param[out] output Output array of ECEF vectors for user motion
 *  \param[in] filename File name of the text input file
 *  \returns Number of user data motion records read, -1 on error
 */
int read_user_motion(const std::span<vec3, USER_MOTION_SIZE> output, const std::string &filename) {
    std::ifstream fs{filename};
    if (!fs.is_open()) {
        return -1;
    }

    int         num_read;
    std::string line;
    for (num_read = 0; num_read < USER_MOTION_SIZE; num_read++) {
        if (!std::getline(fs, line)) {
            break;
        }

        // Read CSV line
        const auto result = scn::scan<double, double, double, double>(line, "{},{},{},{}");
        if (!result) {
            break;
        }
        const auto [t, x, y, z] = result->values();

        output[num_read] = vec3{x, y, z};
    }

    return num_read;
}

/*! \brief Read the list of user motions from the input file
 *  \param[out] output Output array of LatLonHei coordinates for user motion
 *  \param[in] filename File name of the text input file with format Lat,Lon,Hei
 *  \returns Number of user data motion records read, -1 on error
 *
 * Added by romalvarezllorens@gmail.com
 */
int read_user_motion_llh(const std::span<vec3, USER_MOTION_SIZE> output, const std::string &filename) {
    std::ifstream fs{filename};
    if (!fs.is_open()) {
        return -1;
    }

    for (int numd = 0; numd < USER_MOTION_SIZE; ++numd) {
        std::string line;
        if (!std::getline(fs, line)) {
            return numd;
        }

        // Read CSV line
        const auto result = scn::scan<double, double, double, double>(line, "{},{},{},{}");
        if (!result) {
            return numd;
        }
        const auto [t, lat, lon, hei] = result->values();

        if (lat > 90.0 || lat < -90.0 || lon > 180.0 || lon < -180.0) {
            std::cerr << "ERROR: Invalid file format (time[s], latitude[deg], longitude[deg], height [m].\n";
            return 0; // Empty user motion
        }

        const auto llh = vec3{
            lat / R2D, // convert to RAD
            lon / R2D, // convert to RAD
            hei};

        output[numd] = llh2xyz(llh);
    }

    return USER_MOTION_SIZE;
}

std::vector<std::string> string_split(const std::string &s, char delimiter) {
    std::vector<std::string> result;

    for (auto subrange : s | views::split(delimiter)) {
        result.emplace_back(subrange.begin(), subrange.end());
    }

    return result;
}

int read_nmea_gga(const std::span<vec3, USER_MOTION_SIZE> output, const std::string &filename) {
    std::ifstream fs{filename};
    if (!fs.is_open()) {
        return -1;
    }

    std::string line;
    for (int numd = 0; numd < USER_MOTION_SIZE; ++numd) {
        if (!std::getline(fs, line)) {
            return numd;
        }

        const auto elements = string_split(line, ',');
        if (elements.size() != 15) {
            return 0;
        }
        auto it = elements.begin();

        if (std::string_view token = *it++; token.substr(3, 3) == "GGA") {
            token = *it++; // Date and time

            token = *it++; // Latitude
            vec3   llh;
            double temp;
            from_chars(token.substr(0, 2), temp);
            from_chars(token.substr(2), llh.x);
            llh.x = temp + llh.x / 60.0;

            token = *it++; // North or south
            if (token == "S") {
                llh.x *= -1.0;
            }

            llh.x /= R2D; // in radian

            token = *it++; // Longitude
            from_chars(token.substr(0, 3), temp);
            from_chars(token.substr(3), llh.y);
            llh.y = temp + llh.y / 60.0;

            token = *it++; // East or west
            if (token == "W") {
                llh.y *= -1.0;
            }

            llh.y /= R2D; // in radian

            token = *it++; // GPS fix
            token = *it++; // Number of satellites
            token = *it++; // HDOP

            token = *it++; // Altitude above meas sea level
            from_chars(token, llh.z);

            token = *it++; // in meter

            token = *it++; // Geoid height above WGS84 ellipsoid
            from_chars(token, temp);
            llh.z += temp;

            // Convert geodetic position into ECEF coordinates
            output[numd] = llh2xyz(llh);
        }
    }

    return USER_MOTION_SIZE;
}

void generate_nav_msg(const gpstime_t g, channel_t &chan, const bool init) {
    // Data bit reference time
    chan.g0 = gpstime_t{
        .week = g.week,
        .sec  = std::floor(std::round(g.sec) / 30.0) * 30.0 // Align with the full frame length = 30 sec
    };

    auto tow = static_cast<unsigned long>(chan.g0.sec) / 6UL;

    auto prev_word = 0UL;
    if (init) { // Initialize subframe 5
        for (size_t i = 0; i < N_DWORD_SBF; i++) {
            auto sbf_word = chan.sbf[4][i];

            // Add TOW-count message into HOW
            if (i == 1) {
                sbf_word |= (tow & 0x1FFFFUL) << 13;
            }

            // Compute checksum
            sbf_word |= prev_word << 30 & 0xC0000000UL; // 2 LSBs of the previous transmitted word
            const auto nib = i == 1 || i == 9;          // Non-information bearing bits for word 2 and 10
            chan.dwrd[i]   = compute_checksum(sbf_word, nib);

            prev_word = chan.dwrd[i];
        }
    } else { // Save subframe 5
        for (size_t i = 0; i < N_DWORD_SBF; i++) {
            chan.dwrd[i] = chan.dwrd[N_DWORD_SBF * N_SBF + i];
        }
        prev_word = chan.dwrd[N_DWORD_SBF - 1];
        /*
        // Sanity check
        if ((chan.dwrd[1] & 0x1FFFFUL << 13) != (tow & 0x1FFFFUL) << 13) {
            std::cerr << "\nWARNING: Invalid TOW in subframe 5.\n";
            return;
        }
        */
    }

    const auto wn = static_cast<unsigned long>(chan.g0.week % 1024);

    for (size_t i_sbf = 0; i_sbf < N_SBF; i_sbf++) {
        tow++;

        for (size_t i_word = 0; i_word < N_DWORD_SBF; i_word++) {
            unsigned sbf_word = chan.sbf[i_sbf][i_word];

            // Add transmission week number to Subframe 1
            if (i_sbf == 0 && i_word == 2) {
                sbf_word |= (wn & 0x3FFUL) << 20;
            }

            // Add TOW-count message into HOW
            if (i_word == 1) {
                sbf_word |= (tow & 0x1FFFFUL) << 13;
            }

            // Compute checksum
            sbf_word |= prev_word << 30 & 0xC0000000UL;  // 2 LSBs of the previous transmitted word
            const auto nib = i_word == 1 || i_word == 9; // Non-information bearing bits for word 2 and 10
            chan.dwrd[(i_sbf + 1) * N_DWORD_SBF + i_word] = compute_checksum(sbf_word, nib);

            prev_word = chan.dwrd[(i_sbf + 1) * N_DWORD_SBF + i_word];
        }
    }
}

bool check_sat_visibility(
    const ephem_t &eph, const gpstime_t &g, const vec3 &xyz, const double elv_mask, const std::span<double, 2> azel) {
    if (!eph.valid) { // Invalid
        return false;
    }

    const auto llh = xyz2llh(xyz);
    double     tmat[3][3];
    ltcmat(llh, tmat);

    vec3                  pos, vel;
    std::array<double, 2> clk;
    satpos(eph, g, pos, vel, clk);

    const auto los = pos - xyz;
    const auto neu = ecef2neu(los, tmat);
    neu2azel(neu, azel);

    // Visibility
    return azel[1] * R2D > elv_mask;
}

size_t allocate_channel(
    const std::span<channel_t, MAX_CHAN>    chan,
    const std::span<const ephem_t, MAX_SAT> eph,
    const ionoutc_t                        &ionoutc,
    const gpstime_t                        &grx,
    const vec3                             &xyz) {

    size_t sat_count = 0;

    for (size_t sv = 0; sv < MAX_SAT; ++sv) {
        std::array<double, 2> azel;
        if (!check_sat_visibility(eph[sv], grx, xyz, 0.0, azel) && allocated_sat[sv] >= 0) { // Not visible but allocated
            // Clear channel
            chan[allocated_sat[sv]].prn = 0;

            // Clear satellite allocation flag
            allocated_sat[sv] = -1;
            continue;
        }

        ++sat_count; // Number of visible satellites

        if (allocated_sat[sv] != -1) {
            continue;
        }

        // Visible but not allocated

        // Allocated new satellite
        int i;
        for (i = 0; i < MAX_CHAN; i++) {
            if (chan[i].prn != 0) {
                continue;
            }

            // Initialize channel
            chan[i].prn     = static_cast<int>(sv + 1);
            chan[i].azel[0] = azel[0];
            chan[i].azel[1] = azel[1];

            // C/A code generation
            codegen(chan[i].ca, chan[i].prn);

            // Generate subframe
            eph2sbf(eph[sv], ionoutc, chan[i].sbf);

            // Generate navigation message
            generate_nav_msg(grx, chan[i], true);

            // Initialize pseudorange
            range_t rho;
            compute_range(rho, eph[sv], ionoutc, grx, xyz);
            chan[i].rho0 = rho;

            // Initialize carrier phase
            const auto r_xyz = rho.range;

            constexpr vec3 ref;
            compute_range(rho, eph[sv], ionoutc, grx, ref);
            const auto r_ref = rho.range;

            auto phase_ini = 0.0; // TODO: Must initialize properly
                                  // phase_ini = (2.0*r_ref - r_xyz)/LAMBDA_L1;
#ifdef FLOAT_CARR_PHASE
            chan[i].carr_phase = phase_ini - floor(phase_ini);
#else
            phase_ini -= floor(phase_ini);
            chan[i].carr_phase = static_cast<unsigned int>(512.0 * 65536.0 * phase_ini);
#endif
            // Done.
            break;
        }

        // Set satellite allocation channel
        if (i < MAX_CHAN) {
            allocated_sat[sv] = i;
        }
    }

    return sat_count;
}

template <typename T = std::string>
requires std::is_same_v<T, std::string> || std::is_arithmetic_v<T>
auto opt() {
    return cxxopts::value<T>();
}

args_t parse_args(const int argc, char *argv[]) {
    const auto duration_desc = std::format(
        "Duration [sec] (dynamic mode max: {:.0f}, static mode max: {})",
        static_cast<double>(USER_MOTION_SIZE) / 10.0,
        STATIC_MAX_DURATION);

    // clang-format off
    cxxopts::Options options{"gps-sdr-sim", "GPS Software Defined Radio Signal Simulator"};
    options.add_options()
        ("e", "RINEX navigation file for GPS ephemerides (required)", opt(), "<gps_nav>")
        ("u", "User motion file in ECEF x, y, z format (dynamic mode)", opt(), "<user_motion>")
        ("x", "User motion file in lat, lon, height format (dynamic mode)", opt(), "<user_motion>")
        ("g", "NMEA GGA stream (dynamic mode)", opt(), "<nmea_gga>")
        ("c", "ECEF X,Y,Z in meters (static mode) e.g. 3967283.154,1022538.181,4872414.484", opt(), "<location>")
        ("l", "Lat, lon, height (static mode) e.g. 35.681298,139.766247,10.0", opt(), "<location>")
        ("L", "User leap future event in GPS week number, day number, next leap second e.g. 2347,3,19", opt(), "<wnslf,dn,dtslf>")
        ("t", "Scenario start time YYYY/MM/DD,hh:mm:ss", opt(), "<date,time>")
        ("T", "Overwrite TOC and TOE to scenario start time", opt(), "<date,time>")
        ("d", duration_desc, opt<double>(), "<duration>")
        ("o", "I/Q sampling data file", opt()->default_value("gpssim.bin"), "<output>")
        ("s", "Sampling frequency [Hz]", opt<double>()->default_value("2600000"), "<frequency>")
        ("b", "I/Q data format [1/8/16]", opt<int>()->default_value("16"), "<iq_bits>")
        ("i", "Disable ionospheric delay for spacecraft scenario")
        ("p", "Disable path loss and hold power level constant", cxxopts::value<int>()->implicit_value("128"), "[fixed_gain]")
        ("v", "Show details about simulated channels")
        ("h", "Print this help message", cxxopts::value<std::string>());
    // clang-format on

    args_t args;
    if (argc < 3) {
        std::cerr << options.help() << '\n';
        args.valid = false;
        return args;
    }

    const auto parse_result = options.parse(argc, argv);
    const std::unordered_map<std::string, std::function<bool(args_t & args, const cxxopts::KeyValue &option)>> option_handlers{
        {"e",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.nav_file = option.value();
             return true;
         }},
        {"u",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.um_file  = option.value();
             args.nmea_gga = false;
             args.um_llh   = false;
             return true;
         }},
        {"x",
         [](args_t &args, const cxxopts::KeyValue &option) {
             // Added by romalvarezllorens@gmail.com
             args.um_file = option.value();
             args.um_llh  = true;
             return true;
         }},
        {"g",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.um_file  = option.value();
             args.nmea_gga = true;
             return true;
         }},
        {"c",
         [](args_t &args, const cxxopts::KeyValue &option) {
             // Static ECEF coordinates input mode
             args.static_location_mode = true;

             const auto result = scn::scan<double, double, double>(option.value(), "{},{},{}");
             if (!result) {
                 std::cerr << "ERROR: Invalid location format.\n";
                 return false;
             }

             const auto [x, y, z] = result->values();
             xyz[0]               = vec3{x, y, z};
             return true;
         }},
        {"l",
         [](args_t &args, const cxxopts::KeyValue &option) {
             // Static geodetic coordinates input mode
             // Added by scateu@gmail.com
             args.static_location_mode = true;

             const auto result = scn::scan<double, double, double>(option.value(), "{},{},{}");
             if (!result) {
                 std::cerr << "ERROR: Invalid location format.\n";
                 return false;
             }

             const auto [x, y, z] = result->values();
             args.llh             = vec3{
                 x / R2D, // convert to RAD
                 y / R2D, // convert to RAD
                 z};
             xyz[0] = llh2xyz(args.llh); // Convert llh to xyz
             return true;
         }},
        {"L",
         [](args_t &args, const cxxopts::KeyValue &option) {
             // enable custom Leap Event
             auto &ionoutc  = args.ionoutc;
             ionoutc.leapen = true;

             const auto result = scn::scan<int, int, int>(option.value(), "{},{},{}");
             if (!result) {
                 std::cerr << "ERROR: Invalid leap future event format.\n";
                 return false;
             }
             const auto [wnlsf, dn, dtlsf] = result->values();

             if (dn < 1 || dn > 7) {
                 std::cerr << "ERROR: Invalid GPS day number\n";
                 return false;
             }

             if (wnlsf < 0) {
                 std::cerr << "ERROR: Invalid GPS week number\n";
                 return false;
             }

             if (dtlsf < -128 || dtlsf > 127) {
                 std::cerr << "ERROR: Invalid delta leap second\n";
                 return false;
             }

             ionoutc.wnlsf = wnlsf;
             ionoutc.dn    = dn;
             ionoutc.dtlsf = dtlsf;

             return true;
         }},
        {"t",
         [](args_t &args, const cxxopts::KeyValue &option) {
             const auto result = scn::scan<int, int, int, int, int, double>(option.value(), "{}/{}/{},{}:{}:{}");
             if (!result) {
                 std::cerr << "ERROR: Invalid date and time format.\n";
                 return false;
             }
             const auto [y, m, d, hh, mm, sec] = result->values();

             if (y <= 1980 || m < 1 || m > 12 || d < 1 || d > 31 || hh < 0 || hh > 23 || mm < 0 || mm > 59 || sec < 0.0 ||
                 sec >= 60.0) {
                 std::cerr << "ERROR: Invalid date and time.\n";
                 return false;
             }

             args.t0 = datetime_t{.y = y, .m = m, .d = d, .hh = hh, .mm = mm, .sec = std::floor(sec)};
             args.g0 = date2gps(args.t0);

             return true;
         }},
        {"T",
         [&](args_t &args, const cxxopts::KeyValue &option) {
             args.time_overwrite = true;
             if (option.value() == "now") {
                 const auto                   now     = chrono::utc_clock::now();
                 const auto                   sys_now = chrono::clock_cast<chrono::system_clock>(now);
                 const auto                   days    = chrono::floor<chrono::days>(sys_now);
                 const chrono::year_month_day ymd{days};
                 const chrono::hh_mm_ss       hms{sys_now - days};
                 const auto                   seconds = hms.seconds() + hms.subseconds();

                 args.t0 = datetime_t{
                     .y   = static_cast<int>(ymd.year()),
                     .m   = static_cast<int>(static_cast<uint32_t>(ymd.month())),
                     .d   = static_cast<int>(static_cast<uint32_t>(ymd.day())),
                     .hh  = hms.hours().count(),
                     .mm  = hms.minutes().count(),
                     .sec = chrono::duration_cast<chrono::duration<double>>(seconds).count()};

                 args.g0 = date2gps(args.t0);

                 return true;
             }

             return option_handlers.at("t")(args, option);
         }},
        {"d",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.duration = option.as<double>();
             return true;
         }},
        {"o",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.out_file = option.value();
             return true;
         }},
        {"s",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.sampling_frequency = option.as<double>();
             if (args.sampling_frequency < 1.0e6) {
                 std::cerr << "ERROR: Invalid sampling frequency.\n";
                 return false;
             }

             return true;
         }},
        {"b",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.data_format = option.as<int>();
             if (args.data_format != SC01 && args.data_format != SC08 && args.data_format != SC16) {
                 std::cerr << "ERROR: Invalid I/Q data format.\n";
                 return false;
             }

             return true;
         }},
        {"i",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.ionoutc.enable = false; // Disable ionospheric correction
             return true;
         }},
        {"p",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.fixed_gain = option.as<int>();
             if (args.fixed_gain < 1 || args.fixed_gain > 128) {
                 std::cerr << "ERROR: Fixed gain must be between 1 and 128.\n";
                 return false;
             }
             args.path_loss_enable = false; // Disable path loss

             return true;
         }},
        {"v",
         [](args_t &args, const cxxopts::KeyValue &option) {
             args.verbose = true;
             return true;
         }},
        // No need to handle 'h' flags, it's handled separately
    };

    if (parse_result.count("h")) {
        std::cerr << options.help() << '\n';
        args.valid = false;
        return args;
    }

    args.valid = true;
    for (const auto &option : parse_result) {
        const auto &key = option.key();
        if (!option_handlers.contains(key)) {
            std::cerr << std::format("ERROR: Unknown option '{}'\n", key);
        }

        if (!option_handlers.at(key)(args, option)) {
            args.valid = false;
            return args;
        }
    }

    return args;
}

} // namespace

int main(int argc, char *argv[]) {
    ephem_t eph[EPHEM_ARRAY_SIZE][MAX_SAT];

    ////////////////////////////////////////////////////////////
    // Read options
    ////////////////////////////////////////////////////////////

    auto args = parse_args(argc, argv);
    if (!args.valid) {
        return 1;
    }

    if (args.nav_file.empty()) {
        std::cerr << "ERROR: GPS ephemeris file is not specified.\n";
        return 1;
    }

    vec3 llh;
    if (args.um_file.empty() && !args.static_location_mode) {
        // Default static location; Tokyo
        args.static_location_mode = true;
        llh                       = vec3{
            35.681298 / R2D, //
            139.766247 / R2D,
            10.0};
    }

    if (args.duration < 0.0 || (args.duration > static_cast<double>(USER_MOTION_SIZE) / 10.0 && !args.static_location_mode) ||
        (args.duration > STATIC_MAX_DURATION && args.static_location_mode)) {
        std::cerr << "ERROR: Invalid duration.\n";
        return 1;
    }

    // Buffer size
    args.sampling_frequency = std::floor(args.sampling_frequency / 10.0);
    const auto iq_buff_size = static_cast<int>(args.sampling_frequency); // samples per 0.1sec
    args.sampling_frequency *= 10.0;

    ////////////////////////////////////////////////////////////
    // Receiver position
    ////////////////////////////////////////////////////////////

    int n_umd;
    if (!args.static_location_mode) {
        // Read user motion file
        if (args.nmea_gga) {
            n_umd = read_nmea_gga(xyz, args.um_file);
        } else if (args.um_llh) {
            n_umd = read_user_motion_llh(xyz, args.um_file);
        } else {
            n_umd = read_user_motion(xyz, args.um_file);
        }

        if (n_umd == -1) {
            std::cerr << "ERROR: Failed to open user motion / NMEA GGA file.\n";
            return 1;
        }
        if (n_umd == 0) {
            std::cerr << "ERROR: Failed to read user motion / NMEA GGA data.\n";
            return 1;
        }

        // Set simulation duration
        const int i_duration = std::lround(args.duration * 10.0);
        n_umd                = std::min(n_umd, i_duration);

        // Set user initial position
        llh = xyz2llh(xyz[0]);
    } else {
        // Static geodetic coordinates input mode: "-l"
        // Added by scateu@gmail.com
        std::cerr << "Using static location mode.\n";

        // Set simulation duration
        const auto i_duration = std::lround(args.duration * 10.0);
        n_umd                 = i_duration;

        // Set user initial position
        xyz[0] = llh2xyz(llh);
    }

    std::cerr << std::format("xyz = {:11.1f}, {:11.1f}, {:11.1f}\n", xyz[0].x, xyz[0].y, xyz[0].z);
    std::cerr << std::format("llh = {:11.6f}, {:11.6f}, {:11.1f}\n", llh.x * R2D, llh.y * R2D, llh.z);

    ////////////////////////////////////////////////////////////
    // Read ephemeris
    ////////////////////////////////////////////////////////////

    auto &ionoutc = args.ionoutc;
    int   n_eph   = read_rinex_nav_all(eph, ionoutc, args.nav_file);

    if (n_eph == 0) {
        std::cerr << "ERROR: No ephemeris available.\n";
        return 1;
    }

    if (n_eph == -1) {
        std::cerr << "ERROR: ephemeris file not found.\n";
        return 1;
    }

    if (args.verbose && ionoutc.valid) {
        std::cerr << std::format(
            "  {:12.3e} {:12.3e} {:12.3e} {:12.3e}\n", ionoutc.alpha0, ionoutc.alpha1, ionoutc.alpha2, ionoutc.alpha3);
        std::cerr << std::format(
            "  {:12.3e} {:12.3e} {:12.3e} {:12.3e}\n", ionoutc.beta0, ionoutc.beta1, ionoutc.beta2, ionoutc.beta3);
        std::cerr << std::format("   {:19.11e} {:19.11e}  {:9d} {:9d}\n", ionoutc.a0, ionoutc.a1, ionoutc.tot, ionoutc.wnt);
        std::cerr << std::format("{:6d}\n", ionoutc.dtls);
    }

    datetime_t t_min;
    gpstime_t  g_min;
    for (size_t sv = 0; sv < MAX_SAT; sv++) {
        if (eph[0][sv].valid) {
            g_min = eph[0][sv].toc;
            t_min = eph[0][sv].t;
            break;
        }
    }

    gpstime_t  g_max{.week = 0, .sec = 0};
    datetime_t t_max{.y = 0, .m = 0, .d = 0, .hh = 0, .mm = 0, .sec = 0};
    for (size_t sv = 0; sv < MAX_SAT; sv++) {
        if (eph[n_eph - 1][sv].valid) {
            g_max = eph[n_eph - 1][sv].toc;
            t_max = eph[n_eph - 1][sv].t;
            break;
        }
    }

    auto &g0 = args.g0;
    auto &t0 = args.t0;
    if (g0.week >= 0) { // Scenario start time has been set.
        if (!args.time_overwrite && (g0 - g_min < 0.0 || g_max - g0 < 0.0)) {
            std::cerr << "ERROR: Invalid start time.\n";
            std::cerr << std::format(
                "t_min = {:4d}/{:02d}/{:02d},{:02d}:{:02d}:{:02.0f} ({}:{:.0f})\n",
                t_min.y,
                t_min.m,
                t_min.d,
                t_min.hh,
                t_min.mm,
                t_min.sec,
                g_min.week,
                g_min.sec);
            std::cerr << std::format(
                "t_max = {:4d}/{:02d}/{:02d},{:02d}:{:02d}:{:02.0f} ({}:{:.0f})\n",
                t_max.y,
                t_max.m,
                t_max.d,
                t_max.hh,
                t_max.mm,
                t_max.sec,
                g_max.week,
                g_max.sec);
            return 1;
        }

        gpstime_t g_tmp{.week = g0.week, .sec = std::floor(g0.sec / 7200.0) * 7200.0};

        // Overwrite the UTC reference week number
        ionoutc.wnt = g_tmp.week;
        ionoutc.tot = static_cast<int>(g_tmp.sec);

        // Iono/UTC parameters may no longer valid
        // ionoutc.valid = FALSE;

        // Overwrite the TOC and TOE to the scenario start time
        const auto d_sec = g_tmp - g_min;
        for (size_t sv = 0; sv < MAX_SAT; sv++) {
            for (int i = 0; i < n_eph; i++) {
                if (!eph[i][sv].valid) {
                    continue;
                }

                g_tmp            = eph[i][sv].toc + d_sec;
                const auto t_tmp = gps2date(g_tmp);
                eph[i][sv].toc   = g_tmp;
                eph[i][sv].t     = t_tmp;

                g_tmp          = eph[i][sv].toe + d_sec;
                eph[i][sv].toe = g_tmp;
            }
        }
    } else {
        g0 = g_min;
        t0 = t_min;
    }

    std::cerr << std::format(
        "Start time = {:4d}/{:02d}/{:02d},{:02d}:{:02d}:{:02.0f} ({}:{:.0f})\n",
        t0.y,
        t0.m,
        t0.d,
        t0.hh,
        t0.mm,
        t0.sec,
        g0.week,
        g0.sec);
    std::cerr << std::format("Duration = {:.1f} [sec]\n", static_cast<double>(n_umd) / 10.0);

    // Select the current set of ephemerides
    int i_eph = -1;

    for (int i = 0; i < n_eph; i++) {
        for (size_t sv = 0; sv < MAX_SAT; sv++) {
            if (!eph[i][sv].valid) {
                continue;
            }

            if (const auto dt = g0 - eph[i][sv].toc; dt >= -SECONDS_IN_HOUR && dt < SECONDS_IN_HOUR) {
                i_eph = i;
                break;
            }
        }

        if (i_eph >= 0) { // i_eph has been set
            break;
        }
    }

    if (i_eph == -1) {
        std::cerr << "ERROR: No current set of ephemerides has been found.\n";
        return 1;
    }

    ////////////////////////////////////////////////////////////
    // Baseband signal buffer and output file
    ////////////////////////////////////////////////////////////

    // Allocate I/Q buffer
    const auto iq_buff = std::make_unique_for_overwrite<int16_t[]>(2ULL * iq_buff_size);

    if (!iq_buff) {
        std::cerr << "ERROR: Failed to allocate 16-bit I/Q buffer.\n";
        return 1;
    }

    std::unique_ptr<int8_t[]> iq8_buff;
    switch (args.data_format) {
    case SC08:
        iq8_buff = std::make_unique<int8_t[]>(2ULL * iq_buff_size);
        if (!iq8_buff) {
            std::cerr << "ERROR: Failed to allocate 8-bit I/Q buffer.\n";
            return 1;
        }
        break;
    case SC01:
        iq8_buff = std::make_unique<int8_t[]>(iq_buff_size / 4); // byte = {I0, Q0, I1, Q1, I2, Q2, I3, Q3}
        if (!iq8_buff) {
            std::cerr << "ERROR: Failed to allocate compressed 1-bit I/Q buffer.\n";
            return 1;
        }
        break;

    default:
        break;
    }

    // Open output file
    // "-" can be used as name for stdout
    std::unique_ptr<std::ofstream> fs_ptr;
    if (args.out_file != "-") {
        fs_ptr = std::make_unique<std::ofstream>(args.out_file, std::ios::binary);
        if (!fs_ptr->is_open()) {
            std::cerr << "ERROR: Failed to open output file.\n";
            return 1;
        }
    }

    std::ostream &os = fs_ptr ? *fs_ptr : std::cout;

    ////////////////////////////////////////////////////////////
    // Initialize channels
    ////////////////////////////////////////////////////////////

    // Clear all channels
    std::array<channel_t, MAX_CHAN> channels;
    for (auto &chan : channels) {
        chan.prn = 0;
    }

    // Clear satellite allocation flag
    ranges::fill(allocated_sat, -1);

    // Initial reception time
    auto grx = g0 + 0.0;

    // Allocate visible satellites
    allocate_channel(channels, eph[i_eph], ionoutc, grx, xyz[0]);

    for (auto &chan : channels) {
        if (chan.prn <= 0) {
            continue;
        }

        std::cerr << std::format(
            "{:02d} {:6.1f} {:5.1f} {:11.1f} {:5.1f}\n",
            chan.prn,
            chan.azel[0] * R2D,
            chan.azel[1] * R2D,
            chan.rho0.d,
            chan.rho0.iono_delay);
    }

    ////////////////////////////////////////////////////////////
    // Receiver antenna gain pattern
    ////////////////////////////////////////////////////////////

    std::array<double, 37> ant_pat;
    for (size_t i = 0; i < ant_pat.size(); ++i) {
        ant_pat[i] = pow(10.0, -ANT_PAT_DB[i] / 20.0);
    }

    ////////////////////////////////////////////////////////////
    // Generate baseband signals
    ////////////////////////////////////////////////////////////

    const auto t_start = chrono::steady_clock::now();

    // Update receiver time
    grx += 0.1;

    const auto delt = 1.0 / args.sampling_frequency;

    std::array<int, MAX_CHAN> gain;
    for (int i_umd = 1; i_umd < n_umd; i_umd++) {
        for (size_t i = 0; i < MAX_CHAN; i++) {
            if (channels[i].prn <= 0) {
                continue;
            }

            // Refresh code phase and data bit counters
            range_t rho;
            size_t  sv = channels[i].prn - 1;

            // Current pseudorange
            compute_range(rho, eph[i_eph][sv], ionoutc, grx, xyz[args.static_location_mode ? 0 : i_umd]);

            channels[i].azel[0] = rho.azel[0];
            channels[i].azel[1] = rho.azel[1];

            // Update code phase and data bit counters
            compute_code_phase(channels[i], rho, 0.1);
#ifndef FLOAT_CARR_PHASE
            channels[i].carr_phasestep = static_cast<int>(round(512.0 * 65536.0 * channels[i].f_carr * delt));
#endif
            // Path loss
            const auto path_loss = 20200000.0 / rho.d;

            // Receiver antenna gain
            const auto ibs      = static_cast<int>((90.0 - rho.azel[1] * R2D) / 5.0); // covert elevation to boresight
            const auto ant_gain = ant_pat[ibs];

            // Signal gain
            gain[i] = args.path_loss_enable ? static_cast<int>(path_loss * ant_gain * 128.0) : args.fixed_gain; // scaled by 2^7
        }

        for (int i_samp = 0; i_samp < iq_buff_size; ++i_samp) {
            int i_acc = 0;
            int q_acc = 0;

            for (int i = 0; i < MAX_CHAN; i++) {
                if (channels[i].prn <= 0) {
                    continue;
                }

#ifdef FLOAT_CARR_PHASE
                const auto i_table = static_cast<int>(floor(channels[i].carr_phase * 512.0));
#else
                const auto i_table = channels[i].carr_phase >> 16 & 0x1ff; // 9-bit index
#endif
                const auto ip = channels[i].dataBit * channels[i].codeCA * COS_TABLE512[i_table] * gain[i];
                const auto qp = channels[i].dataBit * channels[i].codeCA * SIN_TABLE512[i_table] * gain[i];

                // Accumulate for all visible satellites
                i_acc += ip;
                q_acc += qp;

                // Update code phase
                channels[i].code_phase += channels[i].f_code * delt;

                if (channels[i].code_phase >= CA_SEQ_LEN) {
                    channels[i].code_phase -= CA_SEQ_LEN;

                    channels[i].icode++;

                    if (channels[i].icode >= 20) { // 20 C/A codes = 1 navigation data bit
                        channels[i].icode = 0;
                        channels[i].ibit++;

                        if (channels[i].ibit >= 30) { // 30 navigation data bits = 1 word
                            channels[i].ibit = 0;
                            channels[i].iword++;
                            /*
                            if (channels[i].iword >= N_DWORD) {
                                std::cerr << "\nWARNING: Subframe word buffer overflow.\n";
                            }
                            */
                        }

                        // Set new navigation data bit
                        channels[i].dataBit =
                            static_cast<int>(channels[i].dwrd[channels[i].iword] >> (29 - channels[i].ibit) & 0x1UL) * 2 - 1;
                    }
                }

                // Set current code chip
                channels[i].codeCA = channels[i].ca[static_cast<int>(channels[i].code_phase)] * 2 - 1;

                // Update carrier phase
#ifdef FLOAT_CARR_PHASE
                channels[i].carr_phase += channels[i].f_carr * delt;

                if (channels[i].carr_phase >= 1.0) {
                    channels[i].carr_phase -= 1.0;
                } else if (channels[i].carr_phase < 0.0) {
                    channels[i].carr_phase += 1.0;
                }
#else
                channels[i].carr_phase += channels[i].carr_phasestep;
#endif
            }

            // Scaled by 2^7
            i_acc = (i_acc + 64) >> 7;
            q_acc = (q_acc + 64) >> 7;

            // Store I/Q samples into buffer
            iq_buff[i_samp * 2]     = static_cast<int16_t>(i_acc);
            iq_buff[i_samp * 2 + 1] = static_cast<int16_t>(q_acc);
        }

        switch (args.data_format) {
        case SC01:

            for (int i_samp = 0; i_samp < 2 * iq_buff_size; i_samp++) {
                if (i_samp % 8 == 0) {
                    iq8_buff[i_samp / 8] = 0x00;
                }

                iq8_buff[i_samp / 8] |= (iq_buff[i_samp] > 0 ? 0x01 : 0x00) << (7 - i_samp % 8);
            }

            os.write(reinterpret_cast<char *>(iq8_buff.get()), iq_buff_size / 4);
            break;
        case SC08:
            for (int i_samp = 0; i_samp < 2 * iq_buff_size; i_samp++) {
                iq8_buff[i_samp] = iq_buff[i_samp] >> 4; // 12-bit bladeRF -> 8-bit HackRF
                                                         // iq8_buff[i_samp] = iq_buff[i_samp] >> 8; // for PocketSDR
            }

            os.write(reinterpret_cast<char *>(iq8_buff.get()), 2LL * iq_buff_size);
            break;
        case SC16:
        default:
            os.write(reinterpret_cast<char *>(iq_buff.get()), 2LL * 2 * iq_buff_size);
            break;
        }

        //
        // Update navigation message and channel allocation every 30 seconds
        //

        if (const auto i_grx = std::lround(grx.sec * 10.0); i_grx % 300 == 0) { // Every 30 seconds
            // Update navigation message
            for (auto &channel : channels) {
                if (channel.prn > 0) {
                    generate_nav_msg(grx, channel, false);
                }
            }

            // Refresh ephemeris and subframes
            // Quick and dirty fix. Need more elegant way.
            for (size_t sv = 0; sv < MAX_SAT; sv++) {
                if (eph[i_eph + 1][sv].valid) {
                    if (const auto dt = eph[i_eph + 1][sv].toc - grx; dt < SECONDS_IN_HOUR) {
                        i_eph++;

                        for (auto &channel : channels) {
                            // Generate new subframes if allocated
                            if (channel.prn != 0) {
                                eph2sbf(eph[i_eph][channel.prn - 1], ionoutc, channel.sbf);
                            }
                        }
                    }

                    break;
                }
            }

            // Update channel allocation
            allocate_channel(channels, eph[i_eph], ionoutc, grx, xyz[args.static_location_mode ? 0 : i_umd]);

            // Show details about simulated channels
            if (args.verbose) {
                std::cerr << '\n';
                for (auto &channel : channels) {
                    if (channel.prn > 0) {
                        std::cerr << std::format(
                            "{:02d} {:6.1f} {:5.1f} {:11.1f} {:5.1f}\n",
                            channel.prn,
                            channel.azel[0] * R2D,
                            channel.azel[1] * R2D,
                            channel.rho0.d,
                            channel.rho0.iono_delay);
                    }
                }
            }
        }

        // Update receiver time
        grx += 0.1;

        // Update time counter
        std::cerr << std::format("\rTime into run = {:4.1f}", grx - g0);
        os.flush();
        std::cerr.flush();
    }

    const auto t_end = chrono::steady_clock::now();

    std::cerr << "\nDone!\n";

    // Process time
    const auto elapsed = chrono::duration_cast<chrono::duration<double>>(t_end - t_start);
    std::cerr << std::format("Process time = {:.1f} [sec]\n", elapsed.count());

    return 0;
}
