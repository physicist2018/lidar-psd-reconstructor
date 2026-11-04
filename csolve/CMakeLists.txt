cmake_minimum_required(VERSION 3.16)
project(csolve C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

set(CORE_SRC
    src/util.c
    src/spline.c
    src/linalg.c
    src/tables.c
    src/kernels.c
    src/basis.c
    src/design.c
    src/inverse.c
    src/ensemble.c
    src/smoothing.c
    src/profile.c
    src/batch.c
    src/stats.c
    src/json_in.c
    src/json_out.c
    src/report.c
    src/plot.c
    src/cli.c)

add_library(csolve_core STATIC ${CORE_SRC})
target_include_directories(csolve_core PUBLIC include)
target_link_libraries(csolve_core PUBLIC m)

add_executable(csolve src/main.c)
target_link_libraries(csolve PRIVATE csolve_core)

enable_testing()
add_executable(unit_tests tests/unit_tests.c)
target_link_libraries(unit_tests PRIVATE csolve_core)
add_test(NAME unit_tests COMMAND unit_tests)
