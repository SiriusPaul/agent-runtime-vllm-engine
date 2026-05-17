# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("llama_osh26")

if (NOT DEFINED LLAMA_ROOT)
    message(FATAL_ERROR "LLAMA_ROOT must point to the llama.cpp repository root")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_WEBUI OFF CACHE BOOL "" FORCE)

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_VALIDATE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_RUN_TESTS OFF CACHE BOOL "" FORCE)

if (ANDROID AND CMAKE_ANDROID_NDK)
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        set(_OSH26_GLSLC "${CMAKE_ANDROID_NDK}/shader-tools/windows-x86_64/glslc.exe")
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        set(_OSH26_GLSLC "${CMAKE_ANDROID_NDK}/shader-tools/darwin-x86_64/glslc")
    else()
        set(_OSH26_GLSLC "${CMAKE_ANDROID_NDK}/shader-tools/linux-x86_64/glslc")
    endif()
    if (EXISTS "${_OSH26_GLSLC}")
        set(Vulkan_GLSLC_EXECUTABLE "${_OSH26_GLSLC}" CACHE FILEPATH "" FORCE)
    endif()
endif()

add_subdirectory(${LLAMA_ROOT} llama-build)

if (TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE
        ${LLAMA_ROOT}/vendor/vulkan-headers/include
        ${LLAMA_ROOT}/vendor/spirv-headers/include)
endif()

add_library(llama_osh26 SHARED
    native-lib.cpp
    osh26_engine.cpp)

target_compile_features(llama_osh26 PRIVATE cxx_std_17)

target_link_libraries(llama_osh26
    llama
    android
    log)

# Vulkan 1.1+ symbols (e.g. vkGetPhysicalDeviceFeatures2) are resolved at
# runtime via the device's system libvulkan.so. Override NDK's --no-undefined
# to allow these PLT entries.
target_link_options(llama_osh26 PRIVATE
    "SHELL:-Wl,-z,undefs")