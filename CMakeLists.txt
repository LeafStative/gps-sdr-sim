cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC AND POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("gps-sdr-sim")

set(PLATFORM_WINDOWS OFF)
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  set(PLATFORM_WINDOWS ON)
endif()

find_package(scn CONFIG REQUIRED)

add_executable(gps-sdr-sim
  src/gpssim.cpp
  $<$<BOOL:${PLATFORM_WINDOWS}>:src/getopt.c>
  src/vec3.cpp
)
include_directories(gps-sdr-sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gps-sdr-sim PRIVATE
  $<$<NOT:$<BOOL:${PLATFORM_WINDOWS}>>:m>
  scn::scn
)

set_property(TARGET gps-sdr-sim PROPERTY CXX_STANDARD 20)
