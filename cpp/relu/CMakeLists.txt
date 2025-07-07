add_executable(relu_baseline baseline.cu)
target_link_libraries(relu_baseline gflags)

add_executable(relu_optimize optimize.cu)
target_link_libraries(relu_optimize gflags)