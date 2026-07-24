set(PAM biopass_pam)
add_library(${PAM} SHARED
    pam.cc
)

# The PAM module is now lightweight and only needs basic PAM headers
target_link_libraries(${PAM}
    pam
)

# New helper binary that contains the actual heavy ML logic
add_executable(biopass-helper
    helper.cc
)

target_compile_definitions(biopass-helper PRIVATE
    BIOPASS_VERSION="${PROJECT_VERSION}"
)

target_include_directories(biopass-helper PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${FaceDetection_INCLUDE_DIRS}
    ${FaceAntiSpoof_INCLUDE_DIRS}
    ${FaceRecognition_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/face
    ${CMAKE_SOURCE_DIR}/fingerprint
)

target_link_libraries(biopass-helper
    ${ONNXRUNTIME_LIB}
    biopass_core
    biopass_face
    biopass_fingerprint
    spdlog::spdlog_header_only
)

# Resident per-user daemon (issue #152): keeps AuthManager warm between PAM
# calls so repeat auth attempts skip model reload. Optional at runtime --
# pam.cc falls back to biopass-helper's cold-start path if this isn't
# installed/running, so it is safe to ship even before packaging picks it up.
add_executable(biopassd
    daemon.cc
)

target_compile_definitions(biopassd PRIVATE
    BIOPASS_VERSION="${PROJECT_VERSION}"
)

target_include_directories(biopassd PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${FaceDetection_INCLUDE_DIRS}
    ${FaceAntiSpoof_INCLUDE_DIRS}
    ${FaceRecognition_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/face
    ${CMAKE_SOURCE_DIR}/fingerprint
)

target_link_libraries(biopassd
    ${ONNXRUNTIME_LIB}
    biopass_core
    biopass_face
    biopass_fingerprint
    spdlog::spdlog_header_only
)

# Install rules
install(TARGETS biopass-helper biopassd RUNTIME DESTINATION "/usr/bin")
install(FILES biopassd.service biopassd.socket DESTINATION "/usr/lib/systemd/user")
