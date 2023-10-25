include_directories(${cannon_INCDIR})

add_executable(time_mpi_matmat time_mpi_matmat.cpp)

target_link_libraries(time_mpi_matmat cannon)
