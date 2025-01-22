cmake_minimum_required(VERSION 3.10)
project(TinyGPT_lib)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

# 添加调试宏
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

# 设置 C++ 17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Abseil 配置
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL ON)

# 添加第三方库
add_subdirectory(${THIRD_PARTY_DIR}/abseil-cpp abseil-cpp)
add_subdirectory(${THIRD_PARTY_DIR}/re2 re2)

# 获取源文件
file(GLOB_RECURSE TinyGPT_src "./*.cpp" "./*.h")

# 设置编译选项，GCC
set_source_files_properties(${TinyGPT_src} PROPERTIES COMPILE_FLAGS "-Werror -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 创建目标库
add_library(${PROJECT_NAME}
        ${TinyGPT_src}
        ${THIRD_PARTY_DIR}/json11/json11.cpp
)

# 包含头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE
        ${THIRD_PARTY_DIR}/json11
)

# 链接库
# . /opt/intel/oneapi/mkl/latest/env/vars.sh
set(MKL_ARCH intel64)
set(MKL_LINK static)
set(MKL_THREADING intel_thread)
find_package(MKL CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME}
        re2::re2
        MKL::MKL
)

# blas
# find_library(BLAS_LIB NAMES openblas)
# if (BLAS_LIB)
#     target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIB})
# else()
#     message(FATAL_ERROR "BLAS library not found")
# endif()
