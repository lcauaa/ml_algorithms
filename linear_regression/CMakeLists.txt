cmake_minimum_required(VERSION 3.14)
project(linear_regression)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenBLAS and OpenMP
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

# Optional CBLAS include path
find_path(CBLAS_INCLUDE_DIR cblas.h)

# Print debug info
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "CBLAS include directory: ${CBLAS_INCLUDE_DIR}")

# Set external lib paths
set(USEFULFUNCS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/personal_lib/include")
set(USEFULFUNCS_LIB_DIR "${CMAKE_SOURCE_DIR}/personal_lib/lib")

# Source files
set(SOURCES
    main.cpp
    functions.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include paths
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include          # your project headers
    ${USEFULFUNCS_INCLUDE_DIR}           # external C lib headers
    ${CBLAS_INCLUDE_DIR}                 # optional CBLAS headers
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    ${USEFULFUNCS_LIB_DIR}/libusefulfuncs.a  # or .so if you want shared
)

# For shared lib: rpath to find .so at runtime
target_link_options(${PROJECT_NAME} PRIVATE
    -Wl,-rpath,${USEFULFUNCS_LIB_DIR}
)
