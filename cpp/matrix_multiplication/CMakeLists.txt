add_executable(cuda_core cuda_core.cu)
add_executable(tensor_core tensor_core.cu)

add_subdirectory(ampere)
add_subdirectory(hopper)