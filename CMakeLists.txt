cmake_minimum_required (VERSION 3.8)

project ("ImProc")

project (${PROJECT_NAME} LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CUDA_VERBOSE_BUILD ON)

add_executable (ImProc "ImProc.cpp" "ImProc.h" "ImProc.cu")

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "75")

target_include_directories(${PROJECT_NAME} PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${CUDA_INCLUDE_DIRS}
)