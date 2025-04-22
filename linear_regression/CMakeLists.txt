cmake_minimum_required(VERSION 3.14)
project(linear_regression)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force generic CPU instructions for compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Set and find OpenBLAS path
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Find CBLAS headers
find_path(CBLAS_INCLUDE_DIR cblas.h)

# Debug messages
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "CBLAS include directory: ${CBLAS_INCLUDE_DIR}")

# Source files
set(SOURCES
    main.cpp
    functions.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
if(CBLAS_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CBLAS_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
)

