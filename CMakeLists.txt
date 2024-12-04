cmake_minimum_required(VERSION 3.10)

# Set the project name
project(ComputeRasterizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# includes the bundled libs
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/laszip)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/openvr/src)
set(GLM_STATIC_LIBRARY_ENABLE ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/glm/glm)

# libs/glfw is prebuilt for windows only, so use system installed glfw 
find_package(glfw3 REQUIRED)

# opengl
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()
find_package(OpenGL REQUIRED)

# cuda
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)  # Adjust this path to your CUDA installation
endif()
find_package(CUDA REQUIRED)

# Add all cpp files under CMAKE_CURRENT_SOURCE_DIR/src to the SOURCES variable
set(MAIN_SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Framebuffer.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GLTimerQueries.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/OpenVRHelper.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/OrbitControls.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Renderer.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Texture.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/unsuck_platform_specific.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/compute/ComputeLasLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/compute/LasLoaderSparse.cpp
)

set(LIB_SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/glew/glew.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/backends/imgui_impl_opengl3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/implot/implot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/implot/implot_items.cpp
)

set(SOURCES ${MAIN_SRCS} ${LIB_SRCS})

set(INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/glew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/laszip
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/implot
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/json
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/laszip
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/openvr/headers
    ${GLFW3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}")


set(LIBRARIES 
    laszip
    openvr_api
    glfw
    OpenGL::GL
    cuda
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Add the executable
add_executable(ComputeRasterizerMain
    ${SOURCES} 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
# Include directories
target_include_directories(ComputeRasterizerMain PUBLIC ${INCLUDE_DIRS})
# Link the libraries
target_link_libraries(ComputeRasterizerMain ${LIBRARIES})

add_executable(ComputeRasterizerPerf
    ${SOURCES} 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_buildup_perf.cpp
)
target_include_directories(ComputeRasterizerPerf PUBLIC ${INCLUDE_DIRS})
target_link_libraries(ComputeRasterizerPerf ${LIBRARIES})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/modules DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})