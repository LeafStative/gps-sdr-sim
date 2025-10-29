#pragma once

#ifndef GPSSIM_H
#define GPSSIM_H

#include <numbers>

#include "gpstime.h"

// #define FLOAT_CARR_PHASE // For RKT simulation. Higher computational load, but smoother carrier phase.

#define TRUE (1)
#define FALSE (0)

/*! \brief Maximum length of a line in a text file (RINEX, motion) */
constexpr auto MAX_CHAR = 100;

/*! \brief Maximum number of satellites in RINEX file */
constexpr size_t MAX_SAT = 32;

/*! \brief Maximum number of channels we simulate */
constexpr auto MAX_CHAN = 16;

/*! \brief Maximum number of user motion points */
#ifndef USER_MOTION_SIZE
#define USER_MOTION_SIZE (3000) // max duration at 10Hz
#endif

/*! \brief Maximum duration for static mode*/
constexpr auto STATIC_MAX_DURATION = 86400; // second

/*! \brief Number of subframes */
constexpr size_t N_SBF = 5; // 5 subframes per frame

/*! \brief Number of words per subframe */
constexpr size_t N_DWORD_SBF = 10; // 10 word per subframe

/*! \brief Number of words */
constexpr auto N_DWORD = (N_SBF + 1) * N_DWORD_SBF; // Subframe word buffer size

/*! \brief C/A code sequence length */
constexpr size_t CA_SEQ_LEN = 1023;

constexpr auto POW2_M5  = 0.03125;
constexpr auto POW2_M19 = 1.907348632812500e-6;
constexpr auto POW2_M29 = 1.862645149230957e-9;
constexpr auto POW2_M31 = 4.656612873077393e-10;
constexpr auto POW2_M33 = 1.164153218269348e-10;
constexpr auto POW2_M43 = 1.136868377216160e-13;
constexpr auto POW2_M55 = 2.775557561562891e-17;

constexpr auto POW2_M50 = 8.881784197001252e-016;
constexpr auto POW2_M30 = 9.313225746154785e-010;
constexpr auto POW2_M27 = 7.450580596923828e-009;
constexpr auto POW2_M24 = 5.960464477539063e-008;

// Conventional values employed in GPS ephemeris model (ICD-GPS-200)
constexpr auto GM_EARTH    = 3.986005e14;
constexpr auto OMEGA_EARTH = 7.2921151467e-5;
constexpr auto PI          = std::numbers::pi;

constexpr auto WGS84_RADIUS       = 6378137.0;
constexpr auto WGS84_ECCENTRICITY = 0.0818191908426;

constexpr auto R2D = 57.2957795131;

constexpr auto SPEED_OF_LIGHT = 2.99792458e8;
constexpr auto LAMBDA_L1      = 0.190293672798365;

/*! \brief GPS L1 Carrier frequency */
constexpr auto CARR_FREQ = 1575.42e6;
/*! \brief C/A code frequency */
constexpr auto CODE_FREQ    = 1.023e6;
constexpr auto CARR_TO_CODE = 1.0 / 1540.0;

// Sampling data format
constexpr auto SC01 = 1;
constexpr auto SC08 = 8;
constexpr auto SC16 = 16;

constexpr auto EPHEM_ARRAY_SIZE = 15; // for daily GPS broadcast ephemers file (brdc)

/*! \brief Structure repreenting UTC time */
struct datetime_t {
    int    y;   /*!< Calendar year */
    int    m;   /*!< Calendar month */
    int    d;   /*!< Calendar day */
    int    hh;  /*!< Calendar hour */
    int    mm;  /*!< Calendar minutes */
    double sec; /*!< Calendar seconds */
};

/*! \brief Structure representing ephemeris of a single satellite */
struct ephem_t {
    int        vflg; /*!< Valid Flag */
    datetime_t t;
    gpstime_t  toc;    /*!< Time of Clock */
    gpstime_t  toe;    /*!< Time of Ephemeris */
    int        iodc;   /*!< Issue of Data, Clock */
    int        iode;   /*!< Isuse of Data, Ephemeris */
    double     deltan; /*!< Delta-N (radians/sec) */
    double     cuc;    /*!< Cuc (radians) */
    double     cus;    /*!< Cus (radians) */
    double     cic;    /*!< Correction to inclination cos (radians) */
    double     cis;    /*!< Correction to inclination sin (radians) */
    double     crc;    /*!< Correction to radius cos (meters) */
    double     crs;    /*!< Correction to radius sin (meters) */
    double     ecc;    /*!< e Eccentricity */
    double     sqrta;  /*!< sqrt(A) (sqrt(m)) */
    double     m0;     /*!< Mean anamoly (radians) */
    double     omg0;   /*!< Longitude of the ascending node (radians) */
    double     inc0;   /*!< Inclination (radians) */
    double     aop;
    double     omgdot; /*!< Omega dot (radians/s) */
    double     idot;   /*!< IDOT (radians/s) */
    double     af0;    /*!< Clock offset (seconds) */
    double     af1;    /*!< rate (sec/sec) */
    double     af2;    /*!< acceleration (sec/sec^2) */
    double     tgd;    /*!< Group delay L2 bias */
    int        svhlth;
    int        codeL2;
    // Working variables follow
    double n;       /*!< Mean motion (Average angular velocity) */
    double sq1e2;   /*!< sqrt(1-e^2) */
    double A;       /*!< Semi-major axis */
    double omgkdot; /*!< OmegaDot-OmegaEdot */
};

struct ionoutc_t {
    bool   enable;
    bool   vflag;
    bool   leapen; // enable custom leap event
    int    dtls;
    int    tot;
    int    wnt;
    int    dtlsf;
    int    dn;
    int    wnlsf;
    double alpha0;
    double alpha1;
    double alpha2;
    double alpha3;
    double beta0;
    double beta1;
    double beta2;
    double beta3;
    double a0;
    double a1;
};

struct range_t {
    gpstime_t g;
    double    range; // pseudorange
    double    rate;
    double    d; // geometric distance
    double    azel[2];
    double    iono_delay;
};

/*! \brief Structure representing a Channel */
struct channel_t {
    int    prn;            /*< PRN Number */
    int    ca[CA_SEQ_LEN]; /*< C/A Sequence */
    double f_carr;         /*< Carrier frequency */
    double f_code;         /*< Code frequency */
#ifdef FLOAT_CARR_PHASE
    double carr_phase;
#else
    unsigned int carr_phase;     /*< Carrier phase */
    int          carr_phasestep; /*< Carrier phasestep */
#endif
    double        code_phase;          /*< Code phase */
    gpstime_t     g0;                  /*!< GPS time at start */
    unsigned long sbf[5][N_DWORD_SBF]; /*!< current subframe */
    unsigned long dwrd[N_DWORD];       /*!< Data words of sub-frame */
    int           iword;               /*!< initial word */
    int           ibit;                /*!< initial bit */
    int           icode;               /*!< initial code */
    int           dataBit;             /*!< current data bit */
    int           codeCA;              /*!< current C/A code */
    double        azel[2];
    range_t       rho0;
};

#endif
