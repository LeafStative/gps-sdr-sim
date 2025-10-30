#define _CRT_SECURE_NO_DEPRECATE

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <charconv>
#include <format>
#include <fstream>
#include <iostream>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>

#include <scn/scan.h>

#ifdef _WIN32
#include "getopt.h"
#else
#include <unistd.h>
#endif

#include "gpssim.h"
#include "vec3.h"

namespace ranges = std::ranges;
namespace views  = std::views;

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
void codegen(int *ca, const int prn) {
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
    constexpr auto predicate = [](const auto c) { return !std::isblank(c); };

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

/*! \brief Read Ephemeris data from the RINEX Navigation file */
/*  \param[out] eph Array of Output SV ephemeris data
 *  \param[in] fname File name of the RINEX file
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
void computeRange(range_t *rho, const ephem_t &eph, ionoutc_t *ionoutc, const gpstime_t g, const vec3 &xyz) {
    // SV position at time of the pseudorange observation.
    vec3                  pos, vel;
    std::array<double, 2> clk;
    satpos(eph, g, pos, vel, clk);

    // Receiver to satellite vector and light-time.
    auto         los = pos - xyz;
    const double tau = los.length() / SPEED_OF_LIGHT;

    // Extrapolate the satellite position backwards to the transmission time.
    pos -= vel * tau;

    // Earth rotation correction. The change in velocity can be neglected.
    const double x_rot = pos.x + pos.y * OMEGA_EARTH * tau;
    const double y_rot = pos.y - pos.x * OMEGA_EARTH * tau;
    pos.x              = x_rot;
    pos.y              = y_rot;

    // New observer to satellite vector and satellite range.
    los                = pos - xyz;
    const double range = los.length();
    rho->d             = range;

    // Pseudorange.
    rho->range = range - SPEED_OF_LIGHT * clk[0];

    // Relative velocity of SV and receiver.
    const double rate = vel * los / range;

    // Pseudorange rate.
    rho->rate = rate; // - SPEED_OF_LIGHT*clk[1];

    // Time of application.
    rho->g = g;

    // Azimuth and elevation angles.
    const auto llh = xyz2llh(xyz);

    double tmat[3][3];
    ltcmat(llh, tmat);

    const auto neu = ecef2neu(los, tmat);
    neu2azel(neu, rho->azel);

    // Add ionospheric delay
    rho->iono_delay = ionospheric_delay(*ionoutc, g, llh, rho->azel);
    rho->range += rho->iono_delay;
}

/*! \brief Compute the code phase for a given channel (satellite)
 *  \param chan Channel on which we operate (is updated)
 *  \param[in] rho1 Current range, after \a dt has expired
 *  \param[in dt delta-t (time difference) in seconds
 */
void computeCodePhase(channel_t *chan, const range_t &rho1, const double dt) {

    // Pseudorange rate.
    const double rhorate = (rho1.range - chan->rho0.range) / dt;

    // Carrier and code frequency.
    chan->f_carr = -rhorate / LAMBDA_L1;
    chan->f_code = CODE_FREQ + chan->f_carr * CARR_TO_CODE;

    // Initial code phase and data bit counters.
    const double ms = (chan->rho0.g - chan->g0 + 6.0 - chan->rho0.range / SPEED_OF_LIGHT) * 1000.0;

    int ims          = static_cast<int>(ms);
    chan->code_phase = (ms - static_cast<double>(ims)) * CA_SEQ_LEN; // in chip

    chan->iword = ims / 600; // 1 word = 30 bits = 600 ms
    ims -= chan->iword * 600;

    chan->ibit = ims / 20; // 1 bit = 20 code = 20 ms
    ims -= chan->ibit * 20;

    chan->icode = ims; // 1 code = 1 ms

    chan->codeCA  = chan->ca[static_cast<int>(chan->code_phase)] * 2 - 1;
    chan->dataBit = static_cast<int>(chan->dwrd[chan->iword] >> (29 - chan->ibit) & 0x1UL) * 2 - 1;

    // Save current pseudorange
    chan->rho0 = rho1;
}

/*! \brief Read the list of user motions from the input file
 *  \param[out] xyz Output array of ECEF vectors for user motion
 *  \param[in] filename File name of the text input file
 *  \returns Number of user data motion records read, -1 on error
 */
int read_user_motion(std::array<vec3, USER_MOTION_SIZE> &xyz, const std::string &filename) {
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

        xyz[num_read] = vec3{x, y, z};
    }

    return num_read;
}

/*! \brief Read the list of user motions from the input file
 *  \param[out] xyz Output array of LatLonHei coordinates for user motion
 *  \param[in] filename File name of the text input file with format Lat,Lon,Hei
 *  \returns Number of user data motion records read, -1 on error
 *
 * Added by romalvarezllorens@gmail.com
 */
int readUserMotionLLH(std::array<vec3, USER_MOTION_SIZE> &xyz, const char *filename) {
    FILE *fp = fopen(filename, "rt");
    if (fp == nullptr) {
        return -1;
    }

    int  numd;
    char str[MAX_CHAR];
    for (numd = 0; numd < USER_MOTION_SIZE; numd++) {
        if (fgets(str, MAX_CHAR, fp) == nullptr) {
            break;
        }

        double t;
        vec3   llh;
        if (EOF == sscanf(str, "%lf,%lf,%lf,%lf", &t, &llh.x, &llh.y, &llh.z)) { // Read CSV line
            break;
        }

        if (llh.x > 90.0 || llh.x < -90.0 || llh.y > 180.0 || llh.y < -180.0) {
            std::cerr << "ERROR: Invalid file format (time[s], latitude[deg], longitude[deg], height [m].\n";
            numd = 0; // Empty user motion
            break;
        }

        llh.x /= R2D; // convert to RAD
        llh.y /= R2D; // convert to RAD

        xyz[numd] = llh2xyz(llh);
    }

    fclose(fp);

    return numd;
}

int readNmeaGGA(std::array<vec3, USER_MOTION_SIZE> &xyz, const char *filename) {
    FILE *fp;
    int   numd = 0;
    char  str[MAX_CHAR];
    vec3  llh;
    char  tmp[8];

    if (nullptr == (fp = fopen(filename, "rt"))) return -1;

    while (true) {
        if (fgets(str, MAX_CHAR, fp) == nullptr) break;

        const char *token = strtok(str, ",");

        if (strncmp(token + 3, "GGA", 3) == 0) {
            token = strtok(nullptr, ","); // Date and time

            token = strtok(nullptr, ","); // Latitude
            strncpy(tmp, token, 2);
            tmp[2] = 0;

            llh.x = atof(tmp) + atof(token + 2) / 60.0;

            token = strtok(nullptr, ","); // North or south
            if (token[0] == 'S') llh.x *= -1.0;

            llh.x /= R2D; // in radian

            token = strtok(nullptr, ","); // Longitude
            strncpy(tmp, token, 3);
            tmp[3] = 0;

            llh.y = atof(tmp) + atof(token + 3) / 60.0;

            token = strtok(nullptr, ","); // East or west
            if (token[0] == 'W') llh.y *= -1.0;

            llh.y /= R2D; // in radian

            token = strtok(nullptr, ","); // GPS fix
            token = strtok(nullptr, ","); // Number of satellites
            token = strtok(nullptr, ","); // HDOP

            token = strtok(nullptr, ","); // Altitude above meas sea level

            llh.z = atof(token);

            token = strtok(nullptr, ","); // in meter

            token = strtok(nullptr, ","); // Geoid height above WGS84 ellipsoid

            llh.z += atof(token);

            // Convert geodetic position into ECEF coordinates
            xyz[numd] = llh2xyz(llh);

            // Update the number of track points
            numd++;

            if (numd >= USER_MOTION_SIZE) break;
        }
    }

    fclose(fp);

    return numd;
}

int generateNavMsg(const gpstime_t g, channel_t *chan, const int init) {
    // int           iwrd;
    // unsigned      sbfwrd;
    // unsigned long prevwrd;
    // int           nib;

    gpstime_t g0;
    g0.week = g.week;
    g0.sec =
        static_cast<double>(static_cast<unsigned long>(g.sec + 0.5) / 30UL) * 30.0; // Align with the full frame length = 30 sec
    chan->g0 = g0;                                                                  // Data bit reference time

    const unsigned long wn  = static_cast<unsigned long>(g0.week % 1024);
    unsigned long       tow = static_cast<unsigned long>(g0.sec) / 6UL;

    unsigned long prevwrd;
    if (init == 1) { // Initialize subframe 5
        prevwrd = 0UL;

        for (size_t iwrd = 0; iwrd < N_DWORD_SBF; iwrd++) {
            unsigned sbfwrd = chan->sbf[4][iwrd];

            // Add TOW-count message into HOW
            if (iwrd == 1) {
                sbfwrd |= (tow & 0x1FFFFUL) << 13;
            }

            // Compute checksum
            sbfwrd |= prevwrd << 30 & 0xC0000000UL;            // 2 LSBs of the previous transmitted word
            int nib          = iwrd == 1 || iwrd == 9 ? 1 : 0; // Non-information bearing bits for word 2 and 10
            chan->dwrd[iwrd] = compute_checksum(sbfwrd, nib);

            prevwrd = chan->dwrd[iwrd];
        }
    } else { // Save subframe 5
        for (size_t iwrd = 0; iwrd < N_DWORD_SBF; iwrd++) {
            chan->dwrd[iwrd] = chan->dwrd[N_DWORD_SBF * N_SBF + iwrd];

            prevwrd = chan->dwrd[iwrd];
        }
        /*
        // Sanity check
        if (((chan->dwrd[1])&(0x1FFFFUL<<13)) != ((tow&0x1FFFFUL)<<13))
        {
                std::cerr << "\nWARNING: Invalid TOW in subframe 5.\n";
                return(0);
        }
        */
    }

    for (size_t isbf = 0; isbf < N_SBF; isbf++) {
        tow++;

        for (size_t iwrd = 0; iwrd < N_DWORD_SBF; iwrd++) {
            unsigned sbfwrd = chan->sbf[isbf][iwrd];

            // Add transmission week number to Subframe 1
            if (isbf == 0 && iwrd == 2) sbfwrd |= (wn & 0x3FFUL) << 20;

            // Add TOW-count message into HOW
            if (iwrd == 1) sbfwrd |= (tow & 0x1FFFFUL) << 13;

            // Compute checksum
            sbfwrd |= prevwrd << 30 & 0xC0000000UL;   // 2 LSBs of the previous transmitted word
            int nib = iwrd == 1 || iwrd == 9 ? 1 : 0; // Non-information bearing bits for word 2 and 10
            chan->dwrd[(isbf + 1) * N_DWORD_SBF + iwrd] = compute_checksum(sbfwrd, nib);

            prevwrd = chan->dwrd[(isbf + 1) * N_DWORD_SBF + iwrd];
        }
    }

    return 1;
}

int checkSatVisibility(
    const ephem_t &eph, const gpstime_t g, const vec3 &xyz, const double elvMask, const std::span<double, 2> azel) {
    double tmat[3][3];

    if (!eph.valid) { // Invalid
        return -1;
    }

    const auto llh = xyz2llh(xyz);
    ltcmat(llh, tmat);

    vec3                  pos, vel;
    std::array<double, 2> clk;
    satpos(eph, g, pos, vel, clk);

    const auto los = pos - xyz;
    const auto neu = ecef2neu(los, tmat);
    neu2azel(neu, azel);

    if (azel[1] * R2D > elvMask) { // Visible
        return 1;
    }

    // else
    return 0; // Invisible
}

int allocateChannel(channel_t *chan, ephem_t *eph, ionoutc_t ionoutc, const gpstime_t grx, const vec3 &xyz, double elvMask) {
    int     nsat = 0;
    double  azel[2];
    range_t rho;

    for (size_t sv = 0; sv < MAX_SAT; sv++) {
        if (checkSatVisibility(eph[sv], grx, xyz, 0.0, azel) == 1) {
            nsat++; // Number of visible satellites

            if (allocated_sat[sv] == -1) { // Visible but not allocated
                int i;
                // Allocated new satellite
                for (i = 0; i < MAX_CHAN; i++) {
                    if (chan[i].prn == 0) {
                        // Initialize channel
                        chan[i].prn     = sv + 1;
                        chan[i].azel[0] = azel[0];
                        chan[i].azel[1] = azel[1];

                        // C/A code generation
                        codegen(chan[i].ca, chan[i].prn);

                        // Generate subframe
                        eph2sbf(eph[sv], ionoutc, chan[i].sbf);

                        // Generate navigation message
                        generateNavMsg(grx, &chan[i], 1);

                        // Initialize pseudorange
                        computeRange(&rho, eph[sv], &ionoutc, grx, xyz);
                        chan[i].rho0 = rho;

                        // Initialize carrier phase
                        double r_xyz = rho.range;

                        constexpr vec3 ref;
                        computeRange(&rho, eph[sv], &ionoutc, grx, ref);
                        double r_ref = rho.range;

                        double phase_ini = 0.0; // TODO: Must initialize properly
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
                }

                // Set satellite allocation channel
                if (i < MAX_CHAN) allocated_sat[sv] = i;
            }
        } else if (allocated_sat[sv] >= 0) { // Not visible but allocated
            // Clear channel
            chan[allocated_sat[sv]].prn = 0;

            // Clear satellite allocation flag
            allocated_sat[sv] = -1;
        }
    }

    return nsat;
}

void usage() {
    std::cerr << std::format(
        "Usage: gps-sdr-sim [options]\n"
        "Options:\n"
        "  -e <gps_nav>        RINEX navigation file for GPS ephemerides (required)\n"
        "  -u <user_motion>    User motion file in ECEF x, y, z format (dynamic mode)\n"
        "  -x <user_motion>    User motion file in lat, lon, height format (dynamic mode)\n"
        "  -g <nmea_gga>       NMEA GGA stream (dynamic mode)\n"
        "  -c <location>       ECEF X,Y,Z in meters (static mode) e.g. 3967283.154,1022538.181,4872414.484\n"
        "  -l <location>       Lat, lon, height (static mode) e.g. 35.681298,139.766247,10.0\n"
        "  -L <wnslf,dn,dtslf> User leap future event in GPS week number, day number, next leap second e.g. 2347,3,19\n"
        "  -t <date,time>      Scenario start time YYYY/MM/DD,hh:mm:ss\n"
        "  -T <date,time>      Overwrite TOC and TOE to scenario start time\n"
        "  -d <duration>       Duration [sec] (dynamic mode max: {:.0f}, static mode max: {})\n"
        "  -o <output>         I/Q sampling data file (default: gpssim.bin)\n"
        "  -s <frequency>      Sampling frequency [Hz] (default: 2600000)\n"
        "  -b <iq_bits>        I/Q data format [1/8/16] (default: 16)\n"
        "  -i                  Disable ionospheric delay for spacecraft scenario\n"
        "  -p [fixed_gain]     Disable path loss and hold power level constant\n"
        "  -v                  Show details about simulated channels\n",
        static_cast<double>(USER_MOTION_SIZE) / 10.0,
        STATIC_MAX_DURATION);
}

} // namespace

int main(int argc, char *argv[]) {
    ephem_t   eph[EPHEM_ARRAY_SIZE][MAX_SAT];
    gpstime_t g0;

    vec3 llh;

    channel_t chan[MAX_CHAN];
    double    elvmask = 0.0; // in degree

    int          ip, qp;
    int          iTable;
    short       *iq_buff  = nullptr;
    signed char *iq8_buff = nullptr;

    gpstime_t grx;
    double    delt;
    int       isamp;

    int  iumd;
    int  numd;
    char umfile[MAX_CHAR];

    int staticLocationMode = FALSE;
    int nmeaGGA            = FALSE;
    int umLLH              = FALSE;

    char navfile[MAX_CHAR];
    char outfile[MAX_CHAR];

    double samp_freq;
    int    iq_buff_size;
    int    data_format;

    int result;

    int    gain[MAX_CHAN];
    double path_loss;
    double ant_gain;
    int    fixed_gain = 128;
    double ant_pat[37];
    int    ibs; // boresight angle index

    datetime_t t0, tmin, tmax;
    gpstime_t  gmin, gmax;
    double     dt;
    int        igrx;

    double duration;
    int    iduration;
    int    verb;

    int timeoverwrite = FALSE; // Overwrite the TOC and TOE in the RINEX file

    ionoutc_t ionoutc;
    int       path_loss_enable = TRUE;

    ////////////////////////////////////////////////////////////
    // Read options
    ////////////////////////////////////////////////////////////

    // Default options
    navfile[0] = 0;
    umfile[0]  = 0;
    strcpy(outfile, "gpssim.bin");
    samp_freq      = 2.6e6;
    data_format    = SC16;
    g0.week        = -1; // Invalid start time
    iduration      = USER_MOTION_SIZE;
    duration       = static_cast<double>(iduration) / 10.0; // Default duration
    verb           = FALSE;
    ionoutc.enable = true;
    ionoutc.leapen = false;

    if (argc < 3) {
        usage();
        return 1;
    }

    while ((result = getopt(argc, argv, "e:u:x:g:c:l:o:s:b:L:T:t:d:ipv")) != -1) {
        switch (result) {
        case 'e':
            strcpy(navfile, optarg);
            break;
        case 'u':
            strcpy(umfile, optarg);
            nmeaGGA = FALSE;
            umLLH   = FALSE;
            break;
        case 'x':
            // Added by romalvarezllorens@gmail.com
            strcpy(umfile, optarg);
            umLLH = TRUE;
            break;
        case 'g':
            strcpy(umfile, optarg);
            nmeaGGA = TRUE;
            break;
        case 'c':
            // Static ECEF coordinates input mode
            staticLocationMode = TRUE;
            sscanf(optarg, "%lf,%lf,%lf", &xyz[0].x, &xyz[0].y, &xyz[0].z);
            break;
        case 'l':
            // Static geodetic coordinates input mode
            // Added by scateu@gmail.com
            staticLocationMode = TRUE;
            sscanf(optarg, "%lf,%lf,%lf", &llh.x, &llh.y, &llh.z);
            llh.x /= R2D;          // convert to RAD
            llh.y /= R2D;          // convert to RAD
            xyz[0] = llh2xyz(llh); // Convert llh to xyz
            break;
        case 'o':
            strcpy(outfile, optarg);
            break;
        case 's':
            samp_freq = atof(optarg);
            if (samp_freq < 1.0e6) {
                std::cerr << "ERROR: Invalid sampling frequency.\n";
                return 1;
            }
            break;
        case 'b':
            data_format = atoi(optarg);
            if (data_format != SC01 && data_format != SC08 && data_format != SC16) {
                std::cerr << "ERROR: Invalid I/Q data format.\n";
                return 1;
            }
            break;
        case 'L':
            // enable custom Leap Event
            ionoutc.leapen = true;
            sscanf(optarg, "%d,%d,%d", &ionoutc.wnlsf, &ionoutc.dn, &ionoutc.dtlsf);
            if (ionoutc.dn < 1 || ionoutc.dn > 7) {
                std::cerr << "ERROR: Invalid GPS day number\n";
                return 1;
            }
            if (ionoutc.wnlsf < 0) {
                std::cerr << "ERROR: Invalid GPS week number\n";
                return 1;
            }
            if (ionoutc.dtlsf < -128 || ionoutc.dtlsf > 127) {
                std::cerr << "ERROR: Invalid delta leap second\n";
                return 1;
            }
            break;
        case 'T':
            timeoverwrite = TRUE;
            if (strncmp(optarg, "now", 3) == 0) {
                time_t timer;
                tm    *gmt;

                time(&timer);
                gmt = gmtime(&timer);

                t0.y   = gmt->tm_year + 1900;
                t0.m   = gmt->tm_mon + 1;
                t0.d   = gmt->tm_mday;
                t0.hh  = gmt->tm_hour;
                t0.mm  = gmt->tm_min;
                t0.sec = static_cast<double>(gmt->tm_sec);

                g0 = date2gps(t0);

                break;
            }
        case 't':
            sscanf(optarg, "%d/%d/%d,%d:%d:%lf", &t0.y, &t0.m, &t0.d, &t0.hh, &t0.mm, &t0.sec);
            if (t0.y <= 1980 || t0.m < 1 || t0.m > 12 || t0.d < 1 || t0.d > 31 || t0.hh < 0 || t0.hh > 23 || t0.mm < 0 ||
                t0.mm > 59 || t0.sec < 0.0 || t0.sec >= 60.0) {
                std::cerr << "ERROR: Invalid date and time.\n";
                return 1;
            }
            t0.sec = floor(t0.sec);
            g0     = date2gps(t0);
            break;
        case 'd':
            duration = atof(optarg);
            break;
        case 'i':
            ionoutc.enable = false; // Disable ionospheric correction
            break;
        case 'p':
            if (optind < argc && argv[optind][0] != '-') { // Check if next item is an argument
                fixed_gain = atoi(argv[optind]);
                if (fixed_gain < 1 || fixed_gain > 128) {
                    std::cerr << "ERROR: Fixed gain must be between 1 and 128.\n";
                    return 1;
                }
                optind++; // Move past this argument for next iteration
            }
            path_loss_enable = FALSE; // Disable path loss
            break;
        case 'v':
            verb = TRUE;
            break;
        case ':':
        case '?':
            usage();
            return 1;
        default:
            break;
        }
    }

    if (navfile[0] == 0) {
        std::cerr << "ERROR: GPS ephemeris file is not specified.\n";
        return 1;
    }

    if (umfile[0] == 0 && !staticLocationMode) {
        // Default static location; Tokyo
        staticLocationMode = TRUE;
        llh.x              = 35.681298 / R2D;
        llh.y              = 139.766247 / R2D;
        llh.z              = 10.0;
    }

    if (duration < 0.0 || (duration > static_cast<double>(USER_MOTION_SIZE) / 10.0 && !staticLocationMode) ||
        (duration > STATIC_MAX_DURATION && staticLocationMode)) {
        std::cerr << "ERROR: Invalid duration.\n";
        return 1;
    }
    iduration = static_cast<int>(duration * 10.0 + 0.5);

    // Buffer size
    samp_freq    = floor(samp_freq / 10.0);
    iq_buff_size = static_cast<int>(samp_freq); // samples per 0.1sec
    samp_freq *= 10.0;

    delt = 1.0 / samp_freq;

    ////////////////////////////////////////////////////////////
    // Receiver position
    ////////////////////////////////////////////////////////////

    if (!staticLocationMode) {
        // Read user motion file
        if (nmeaGGA == TRUE) {
            numd = readNmeaGGA(xyz, umfile);
        } else if (umLLH == TRUE) {
            numd = readUserMotionLLH(xyz, umfile);
        } else {
            numd = read_user_motion(xyz, umfile);
        }

        if (numd == -1) {
            std::cerr << "ERROR: Failed to open user motion / NMEA GGA file.\n";
            return 1;
        }
        if (numd == 0) {
            std::cerr << "ERROR: Failed to read user motion / NMEA GGA data.\n";
            return 1;
        }

        // Set simulation duration
        if (numd > iduration) {
            numd = iduration;
        }

        // Set user initial position
        llh = xyz2llh(xyz[0]);
    } else {
        // Static geodetic coordinates input mode: "-l"
        // Added by scateu@gmail.com
        std::cerr << "Using static location mode.\n";

        // Set simulation duration
        numd = iduration;

        // Set user initial position
        xyz[0] = llh2xyz(llh);
    }

    std::cerr << std::format("xyz = {:11.1f}, {:11.1f}, {:11.1f}\n", xyz[0].x, xyz[0].y, xyz[0].z);
    std::cerr << std::format("llh = {:11.6f}, {:11.6f}, {:11.1f}\n", llh.x * R2D, llh.y * R2D, llh.z);

    ////////////////////////////////////////////////////////////
    // Read ephemeris
    ////////////////////////////////////////////////////////////

    int neph = read_rinex_nav_all(eph, ionoutc, navfile);

    if (neph == 0) {
        std::cerr << "ERROR: No ephemeris available.\n";
        return 1;
    }

    if (neph == -1) {
        std::cerr << "ERROR: ephemeris file not found.\n";
        return 1;
    }

    if (verb == TRUE && ionoutc.valid) {
        std::cerr << std::format(
            "  {:12.3e} {:12.3e} {:12.3e} {:12.3e}\n", ionoutc.alpha0, ionoutc.alpha1, ionoutc.alpha2, ionoutc.alpha3);
        std::cerr << std::format(
            "  {:12.3e} {:12.3e} {:12.3e} {:12.3e}\n", ionoutc.beta0, ionoutc.beta1, ionoutc.beta2, ionoutc.beta3);
        std::cerr << std::format("   {:19.11e} {:19.11e}  {:9d} {:9d}\n", ionoutc.a0, ionoutc.a1, ionoutc.tot, ionoutc.wnt);
        std::cerr << std::format("{:6d}\n", ionoutc.dtls);
    }

    for (size_t sv = 0; sv < MAX_SAT; sv++) {
        if (eph[0][sv].valid) {
            gmin = eph[0][sv].toc;
            tmin = eph[0][sv].t;
            break;
        }
    }

    gmax.sec  = 0;
    gmax.week = 0;
    tmax.sec  = 0;
    tmax.mm   = 0;
    tmax.hh   = 0;
    tmax.d    = 0;
    tmax.m    = 0;
    tmax.y    = 0;
    for (size_t sv = 0; sv < MAX_SAT; sv++) {
        if (eph[neph - 1][sv].valid) {
            gmax = eph[neph - 1][sv].toc;
            tmax = eph[neph - 1][sv].t;
            break;
        }
    }

    if (g0.week >= 0) { // Scenario start time has been set.
        if (timeoverwrite == TRUE) {
            gpstime_t gtmp;

            gtmp.week = g0.week;
            gtmp.sec  = static_cast<double>(static_cast<int>(g0.sec) / 7200) * 7200.0;

            const auto d_sec = gtmp - gmin;

            // Overwrite the UTC reference week number
            ionoutc.wnt = gtmp.week;
            ionoutc.tot = static_cast<int>(gtmp.sec);

            // Iono/UTC parameters may no longer valid
            // ionoutc.valid = FALSE;

            // Overwrite the TOC and TOE to the scenario start time
            for (size_t sv = 0; sv < MAX_SAT; sv++) {
                for (int i = 0; i < neph; i++) {
                    if (eph[i][sv].valid) {
                        gtmp             = eph[i][sv].toc + d_sec;
                        const auto t_tmp = gps2date(gtmp);
                        eph[i][sv].toc   = gtmp;
                        eph[i][sv].t     = t_tmp;

                        gtmp           = eph[i][sv].toe + d_sec;
                        eph[i][sv].toe = gtmp;
                    }
                }
            }
        } else {
            if (g0 - gmin < 0.0 || gmax - g0 < 0.0) {
                std::cerr << "ERROR: Invalid start time.\n";
                std::cerr << std::format(
                    "tmin = {:4d}/{:02d}/{:02d},{:02d}:{:02d}:{:02.0f} ({}:{:.0f})\n",
                    tmin.y,
                    tmin.m,
                    tmin.d,
                    tmin.hh,
                    tmin.mm,
                    tmin.sec,
                    gmin.week,
                    gmin.sec);
                std::cerr << std::format(
                    "tmax = {:4d}/{:02d}/{:02d},{:02d}:{:02d}:{:02.0f} ({}:{:.0f})\n",
                    tmax.y,
                    tmax.m,
                    tmax.d,
                    tmax.hh,
                    tmax.mm,
                    tmax.sec,
                    gmax.week,
                    gmax.sec);
                return 1;
            }
        }
    } else {
        g0 = gmin;
        t0 = tmin;
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
    std::cerr << std::format("Duration = {:.1f} [sec]\n", static_cast<double>(numd) / 10.0);

    // Select the current set of ephemerides
    int ieph = -1;

    for (int i = 0; i < neph; i++) {
        for (size_t sv = 0; sv < MAX_SAT; sv++) {
            if (eph[i][sv].valid) {
                dt = g0 - eph[i][sv].toc;
                if (dt >= -SECONDS_IN_HOUR && dt < SECONDS_IN_HOUR) {
                    ieph = i;
                    break;
                }
            }
        }

        if (ieph >= 0) { // ieph has been set
            break;
        }
    }

    if (ieph == -1) {
        std::cerr << "ERROR: No current set of ephemerides has been found.\n";
        return 1;
    }

    ////////////////////////////////////////////////////////////
    // Baseband signal buffer and output file
    ////////////////////////////////////////////////////////////

    // Allocate I/Q buffer
    iq_buff = static_cast<short *>(calloc(2 * iq_buff_size, 2));

    if (iq_buff == nullptr) {
        std::cerr << "ERROR: Failed to allocate 16-bit I/Q buffer.\n";
        return 1;
    }

    if (data_format == SC08) {
        iq8_buff = static_cast<signed char *>(calloc(2 * iq_buff_size, 1));
        if (iq8_buff == nullptr) {
            std::cerr << "ERROR: Failed to allocate 8-bit I/Q buffer.\n";
            return 1;
        }
    } else if (data_format == SC01) {
        iq8_buff = static_cast<signed char *>(calloc(iq_buff_size / 4, 1)); // byte = {I0, Q0, I1, Q1, I2, Q2, I3, Q3}
        if (iq8_buff == nullptr) {
            std::cerr << "ERROR: Failed to allocate compressed 1-bit I/Q buffer.\n";
            return 1;
        }
    }

    // Open output file
    // "-" can be used as name for stdout
    FILE *fp;
    if (strcmp("-", outfile)) {
        if (nullptr == (fp = fopen(outfile, "wb"))) {
            std::cerr << "ERROR: Failed to open output file.\n";
            return 1;
        }
    } else {
        fp = stdout;
    }

    ////////////////////////////////////////////////////////////
    // Initialize channels
    ////////////////////////////////////////////////////////////

    // Clear all channels
    for (int i = 0; i < MAX_CHAN; i++) {
        chan[i].prn = 0;
    }

    // Clear satellite allocation flag
    for (size_t sv = 0; sv < MAX_SAT; sv++) {
        allocated_sat[sv] = -1;
    }

    // Initial reception time
    grx = g0 + 0.0;

    // Allocate visible satellites
    allocateChannel(chan, eph[ieph], ionoutc, grx, xyz[0], elvmask);

    for (int i = 0; i < MAX_CHAN; i++) {
        if (chan[i].prn > 0)
            std::cerr << std::format(
                "{:02d} {:6.1f} {:5.1f} {:11.1f} {:5.1f}\n",
                chan[i].prn,
                chan[i].azel[0] * R2D,
                chan[i].azel[1] * R2D,
                chan[i].rho0.d,
                chan[i].rho0.iono_delay);
    }

    ////////////////////////////////////////////////////////////
    // Receiver antenna gain pattern
    ////////////////////////////////////////////////////////////

    for (int i = 0; i < 37; i++) {
        ant_pat[i] = pow(10.0, -ANT_PAT_DB[i] / 20.0);
    }

    ////////////////////////////////////////////////////////////
    // Generate baseband signals
    ////////////////////////////////////////////////////////////

    clock_t tstart = clock();

    // Update receiver time
    grx += 0.1;

    for (iumd = 1; iumd < numd; iumd++) {
        for (int i = 0; i < MAX_CHAN; i++) {
            if (chan[i].prn > 0) {
                // Refresh code phase and data bit counters
                range_t rho;
                size_t  sv = chan[i].prn - 1;

                // Current pseudorange
                if (!staticLocationMode) {
                    computeRange(&rho, eph[ieph][sv], &ionoutc, grx, xyz[iumd]);
                } else {
                    computeRange(&rho, eph[ieph][sv], &ionoutc, grx, xyz[0]);
                }

                chan[i].azel[0] = rho.azel[0];
                chan[i].azel[1] = rho.azel[1];

                // Update code phase and data bit counters
                computeCodePhase(&chan[i], rho, 0.1);
#ifndef FLOAT_CARR_PHASE
                chan[i].carr_phasestep = static_cast<int>(round(512.0 * 65536.0 * chan[i].f_carr * delt));
#endif
                // Path loss
                path_loss = 20200000.0 / rho.d;

                // Receiver antenna gain
                ibs      = static_cast<int>((90.0 - rho.azel[1] * R2D) / 5.0); // covert elevation to boresight
                ant_gain = ant_pat[ibs];

                // Signal gain
                if (path_loss_enable == TRUE)
                    gain[i] = static_cast<int>(path_loss * ant_gain * 128.0); // scaled by 2^7
                else
                    gain[i] = fixed_gain; // hold the power level constant
            }
        }

        for (isamp = 0; isamp < iq_buff_size; isamp++) {
            int i_acc = 0;
            int q_acc = 0;

            for (int i = 0; i < MAX_CHAN; i++) {
                if (chan[i].prn > 0) {
#ifdef FLOAT_CARR_PHASE
                    iTable = static_cast<int>(floor(chan[i].carr_phase * 512.0));
#else
                    iTable = chan[i].carr_phase >> 16 & 0x1ff; // 9-bit index
#endif
                    ip = chan[i].dataBit * chan[i].codeCA * COS_TABLE512[iTable] * gain[i];
                    qp = chan[i].dataBit * chan[i].codeCA * SIN_TABLE512[iTable] * gain[i];

                    // Accumulate for all visible satellites
                    i_acc += ip;
                    q_acc += qp;

                    // Update code phase
                    chan[i].code_phase += chan[i].f_code * delt;

                    if (chan[i].code_phase >= CA_SEQ_LEN) {
                        chan[i].code_phase -= CA_SEQ_LEN;

                        chan[i].icode++;

                        if (chan[i].icode >= 20) { // 20 C/A codes = 1 navigation data bit
                            chan[i].icode = 0;
                            chan[i].ibit++;

                            if (chan[i].ibit >= 30) { // 30 navigation data bits = 1 word
                                chan[i].ibit = 0;
                                chan[i].iword++;
                                /*
                                if (chan[i].iword>=N_DWORD)
                                        std::cerr << "\nWARNING: Subframe word buffer overflow.\n";
                                */
                            }

                            // Set new navigation data bit
                            chan[i].dataBit =
                                static_cast<int>(chan[i].dwrd[chan[i].iword] >> (29 - chan[i].ibit) & 0x1UL) * 2 - 1;
                        }
                    }

                    // Set current code chip
                    chan[i].codeCA = chan[i].ca[static_cast<int>(chan[i].code_phase)] * 2 - 1;

                    // Update carrier phase
#ifdef FLOAT_CARR_PHASE
                    chan[i].carr_phase += chan[i].f_carr * delt;

                    if (chan[i].carr_phase >= 1.0)
                        chan[i].carr_phase -= 1.0;
                    else if (chan[i].carr_phase < 0.0)
                        chan[i].carr_phase += 1.0;
#else
                    chan[i].carr_phase += chan[i].carr_phasestep;
#endif
                }
            }

            // Scaled by 2^7
            i_acc = (i_acc + 64) >> 7;
            q_acc = (q_acc + 64) >> 7;

            // Store I/Q samples into buffer
            iq_buff[isamp * 2]     = static_cast<short>(i_acc);
            iq_buff[isamp * 2 + 1] = static_cast<short>(q_acc);
        }

        if (data_format == SC01) {
            for (isamp = 0; isamp < 2 * iq_buff_size; isamp++) {
                if (isamp % 8 == 0) iq8_buff[isamp / 8] = 0x00;

                iq8_buff[isamp / 8] |= (iq_buff[isamp] > 0 ? 0x01 : 0x00) << (7 - isamp % 8);
            }

            fwrite(iq8_buff, 1, iq_buff_size / 4, fp);
        } else if (data_format == SC08) {
            for (isamp = 0; isamp < 2 * iq_buff_size; isamp++) {
                iq8_buff[isamp] = iq_buff[isamp] >> 4; // 12-bit bladeRF -> 8-bit HackRF
                                                       // iq8_buff[isamp] = iq_buff[isamp] >> 8; // for PocketSDR
            }

            fwrite(iq8_buff, 1, 2 * iq_buff_size, fp);
        } else { // data_format==SC16
            fwrite(iq_buff, 2, 2 * iq_buff_size, fp);
        }

        //
        // Update navigation message and channel allocation every 30 seconds
        //

        igrx = static_cast<int>(grx.sec * 10.0 + 0.5);

        if (igrx % 300 == 0) { // Every 30 seconds
            // Update navigation message
            for (int i = 0; i < MAX_CHAN; i++) {
                if (chan[i].prn > 0) generateNavMsg(grx, &chan[i], 0);
            }

            // Refresh ephemeris and subframes
            // Quick and dirty fix. Need more elegant way.
            for (size_t sv = 0; sv < MAX_SAT; sv++) {
                if (eph[ieph + 1][sv].valid) {
                    dt = eph[ieph + 1][sv].toc - grx;
                    if (dt < SECONDS_IN_HOUR) {
                        ieph++;

                        for (int i = 0; i < MAX_CHAN; i++) {
                            // Generate new subframes if allocated
                            if (chan[i].prn != 0) eph2sbf(eph[ieph][chan[i].prn - 1], ionoutc, chan[i].sbf);
                        }
                    }

                    break;
                }
            }

            // Update channel allocation
            if (!staticLocationMode) {
                allocateChannel(chan, eph[ieph], ionoutc, grx, xyz[iumd], elvmask);
            } else {
                allocateChannel(chan, eph[ieph], ionoutc, grx, xyz[0], elvmask);
            }

            // Show details about simulated channels
            if (verb == TRUE) {
                std::cerr << '\n';
                for (int i = 0; i < MAX_CHAN; i++) {
                    if (chan[i].prn > 0)
                        std::cerr << std::format(
                            "{:02d} {:6.1f} {:5.1f} {:11.1f} {:5.1f}\n",
                            chan[i].prn,
                            chan[i].azel[0] * R2D,
                            chan[i].azel[1] * R2D,
                            chan[i].rho0.d,
                            chan[i].rho0.iono_delay);
                }
            }
        }

        // Update receiver time
        grx += 0.1;

        // Update time counter
        std::cerr << std::format("\rTime into run = {:4.1f}", grx - g0);
        fflush(stdout);
    }

    clock_t tend = clock();

    std::cerr << "\nDone!\n";

    // Free I/Q buffer
    free(iq_buff);

    // Close file
    fclose(fp);

    // Process time
    std::cerr << std::format("Process time = {:.1f} [sec]\n", static_cast<double>(tend - tstart) / CLOCKS_PER_SEC);

    return 0;
}
