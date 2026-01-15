# Cloth (Mass-Spring) Simulation

# Source files
set(CLOTH_SOURCES
    main.cu
)

# Create executable
add_executable(cloth_sim ${CLOTH_SOURCES})

# Link libraries
target_link_libraries(cloth_sim
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${OPENGL_LIBRARIES}
    GLUT::GLUT
    ${GLEW_LIBRARIES}
    pthread
    dl
)

# Set CUDA properties
set_target_properties(cloth_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Copy shader files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Square/phong.frag DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Square/phong.vert DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Square/shadow.frag DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Square/shadow.vert DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA_Projective_Square/setting.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
