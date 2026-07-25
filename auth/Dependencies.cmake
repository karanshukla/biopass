# ==============================================================================
# Dependency Management
# ==============================================================================

# ONNX Runtime
set(ONNXRUNTIME_VERSION "1.19.2")
include(FetchContent)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(ONNXRUNTIME_ARCH "linux-aarch64")
else()
    set(ONNXRUNTIME_ARCH "linux-x64")
endif()

FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_ROOT "${onnxruntime_SOURCE_DIR}")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)

# libturbojpeg (system package via pkg-config; stable soname)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TURBOJPEG REQUIRED IMPORTED_TARGET libturbojpeg)

# libcamera (camera capture) is built from source at a pinned version and
# bundled into the package instead of linked against the system's copy -- see
# BundleLibcamera.cmake for why. Provides the `libcamera_bundled` target.
include(${CMAKE_CURRENT_LIST_DIR}/BundleLibcamera.cmake)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        904aa67e1e2d1dec92959df63e700b166d5c1022
)
FetchContent_MakeAvailable(stb)
set(STB_INCLUDE_DIRS "${stb_SOURCE_DIR}")

# yaml-cpp for config parsing
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        yaml-cpp-0.9.0
)
FetchContent_MakeAvailable(yaml-cpp)

# CLI11 for command line parsing
find_package(CLI11 REQUIRED)

# spdlog for logging
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

# sqlite3 (amalgamation, vendored) for read-only access to biopass.db (model
# registry) from the PAM helper. Pinned by hash rather than depending on the
# distro's sqlite3, consistent with the yaml-cpp/spdlog/onnxruntime pinning
# above -- this runs inside a security-sensitive PAM module.
set(SQLITE3_VERSION "3460100")
FetchContent_Declare(
    sqlite3_amalgamation
    URL https://www.sqlite.org/2024/sqlite-amalgamation-${SQLITE3_VERSION}.zip
    URL_HASH SHA256=77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(sqlite3_amalgamation)

add_library(sqlite3 STATIC ${sqlite3_amalgamation_SOURCE_DIR}/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${sqlite3_amalgamation_SOURCE_DIR})
set_target_properties(sqlite3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(sqlite3 PUBLIC
    SQLITE_OMIT_LOAD_EXTENSION
    SQLITE_THREADSAFE=1
    SQLITE_DQS=0
)

# ------------------------------------------------------------------------
# OpenVINO (optional): NPU/GPU inference backend, falling back to the ONNX
# Runtime CPU path above when unavailable at build time or at runtime (no
# matching device on the host, or the device fails to compile the model).
# Off by default -- this is experimental and only tested on Intel NPU
# hardware so far; see onnx_session.cc for the device probe/fallback logic.
#
# Vendored (like ONNX Runtime above) rather than found via the system
# `openvino`/`openvino-devel`/`openvino-plugins` packages: distro packaging
# lags new NPU silicon by multiple OpenVINO releases (Fedora 44 ships
# 2025.1.0, whose NPU plugin speaks an older graph-extension protocol than
# current Intel NPU drivers report -- compile_model() fails outright on
# newer chips against it). Pin a specific upstream release instead of
# tracking "latest" so the NPU backend doesn't silently change behavior
# out from under a build.
#
# No plugins.xml ships in this archive -- modern OpenVINO auto-discovers
# plugins (libopenvino_intel_*_plugin.so, libopenvino_onnx_frontend.so) by
# scanning next to libopenvino.so itself, so no extra wiring is needed
# beyond pointing find_package at the extracted runtime/cmake dir.
# ------------------------------------------------------------------------
option(BIOPASS_USE_OPENVINO "Enable optional OpenVINO NPU/GPU inference backend" OFF)
if(BIOPASS_USE_OPENVINO)
    set(OPENVINO_VERSION "2026.2.1.21919.ede283a88e3")
    FetchContent_Declare(
        openvino
        URL https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2.1/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION}_x86_64.tgz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(openvino)
    find_package(OpenVINO REQUIRED CONFIG
        PATHS "${openvino_SOURCE_DIR}/runtime/cmake"
        NO_DEFAULT_PATH
    )
    set(OPENVINO_LIB_DIR "${openvino_SOURCE_DIR}/runtime/lib/intel64")
    message(STATUS "Biopass: OpenVINO backend enabled (vendored ${OPENVINO_VERSION}, NPU/GPU with ONNX Runtime CPU fallback)")
endif()
