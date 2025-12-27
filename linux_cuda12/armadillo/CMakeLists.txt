# Armadillo (Tetrahedral Mesh) Simulation

# Source files
set(ARMADILLO_SOURCES
    main.cu
)

# Create executable
add_executable(armadillo_sim ${ARMADILLO_SOURCES})

# Link libraries
target_link_libraries(armadillo_sim
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARIES}
    pthread
    dl
)

# Set CUDA properties
set_target_properties(armadillo_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Copy shader files and meshes to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Armadillo/phong.frag DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Armadillo/phong.vert DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Armadillo/shadow.frag DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Armadillo/shadow.vert DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Armadillo/setting.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Armadillo/meshes DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
