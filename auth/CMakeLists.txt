cmake_minimum_required(VERSION 3.22)
# Resolve version from the environment (CI) or -DPROJECT_VERSION (Makefile),
# before project() clears the variable, otherwise fall back to a default.
if(DEFINED ENV{PROJECT_VERSION})
    set(BIOPASS_VERSION $ENV{PROJECT_VERSION})
elseif(PROJECT_VERSION)
    set(BIOPASS_VERSION ${PROJECT_VERSION})
else()
    set(BIOPASS_VERSION 0.1.0)
endif()

project(biopass)
set(PROJECT_VERSION ${BIOPASS_VERSION})
message(STATUS "Project version: ${PROJECT_VERSION}")

# Require C++17
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Avoid RPATH warnings about shadowed libraries in non-system directories
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
# Dependencies Configuration
include(Dependencies.cmake)

# Export Face Includes dynamically from Source
set(FaceDetection_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/face/detection)
set(FaceRecognition_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/face/recognition)
set(FaceAntiSpoof_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/face/antispoofing)

# Application Directories
add_subdirectory(core)
add_subdirectory(face)
add_subdirectory(fingerprint)
add_subdirectory(pam)

# Testing Directories
option(BUILD_TESTS "Build test configurations" OFF)
if(BUILD_TESTS)
    add_subdirectory(test/camera)
    add_subdirectory(test/face_engine)
endif()

install(
    TARGETS biopass_det biopass_reg biopass_as
    COMPONENT applications
    RUNTIME DESTINATION "/usr/bin"
    LIBRARY DESTINATION "/usr/lib/biopass"
)
# PAM modules must land in the directory the system's PAM actually searches,
# which varies by distro/architecture (Fedora/RHEL/SUSE x86_64 use
# /lib64/security; Debian/Ubuntu use the multiarch /lib/<triplet>/security;
# others use /lib/security). Probe for pam_unix.so -- present in every PAM
# install -- to find the real one instead of hardcoding a single distro's path.
# Override with -DPAM_MODULE_DIR=... if detection guesses wrong.
if(NOT DEFINED PAM_MODULE_DIR)
    set(_pam_candidate_dirs "")
    if(CMAKE_LIBRARY_ARCHITECTURE)
        list(APPEND _pam_candidate_dirs
             "/lib/${CMAKE_LIBRARY_ARCHITECTURE}/security"
             "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/security")
    endif()
    list(APPEND _pam_candidate_dirs
         "/lib64/security" "/usr/lib64/security"
         "/lib/security" "/usr/lib/security")
    foreach(_dir IN LISTS _pam_candidate_dirs)
        if(EXISTS "${_dir}/pam_unix.so")
            set(PAM_MODULE_DIR "${_dir}")
            break()
        endif()
    endforeach()
endif()
if(NOT DEFINED PAM_MODULE_DIR)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND IS_DIRECTORY "/lib64")
        set(PAM_MODULE_DIR "/lib64/security")
    else()
        set(PAM_MODULE_DIR "/lib/security")
    endif()
    message(WARNING
        "Biopass: could not locate an existing PAM module dir (pam_unix.so); "
        "defaulting to ${PAM_MODULE_DIR}. Override with -DPAM_MODULE_DIR=... if wrong.")
endif()
message(STATUS "Biopass: installing PAM module to ${PAM_MODULE_DIR}")

install(
    TARGETS biopass_pam
    LIBRARY DESTINATION "${PAM_MODULE_DIR}"
)

# Install ONNX Runtime libraries
install(
    DIRECTORY ${ONNXRUNTIME_LIB_DIR}/
    DESTINATION "/usr/lib/biopass"
    FILES_MATCHING PATTERN "*.so*"
)

# Install OpenVINO runtime + plugin libraries (only when the optional NPU/GPU
# backend is enabled -- see BIOPASS_USE_OPENVINO in Dependencies.cmake).
if(BIOPASS_USE_OPENVINO)
    install(
        DIRECTORY ${OPENVINO_LIB_DIR}/
        DESTINATION "/usr/lib/biopass"
        FILES_MATCHING PATTERN "*.so*"
    )
endif()
