project(logistic_regression)

set(SOURCES
    main.cpp
    logistic_regression_functions.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/personal_lib/include
)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_path(CBLAS_INCLUDE_DIR cblas.h)

target_include_directories(${PROJECT_NAME} PRIVATE ${CBLAS_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    ${CMAKE_SOURCE_DIR}/personal_lib/lib/libusefulfuncs.a
)

target_link_options(${PROJECT_NAME} PRIVATE
    -Wl,-rpath,${CMAKE_SOURCE_DIR}/personal_lib/lib
)
