add_executable(rms_normalization_baseline baseline.cu)
target_link_libraries(rms_normalization_baseline gflags)

add_executable(rms_normalization_warp_shuffle_instruction warp_shuffle_instruction.cu)
target_link_libraries(rms_normalization_warp_shuffle_instruction gflags)