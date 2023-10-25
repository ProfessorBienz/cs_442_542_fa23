include_directories(.)

find_package(CUDAToolkit REQUIRED)

add_library(cannon
    cuda_cannon.cu
    mpi_cannon.cpp
    mpi_cannon.hpp
    utils.cu
)

target_link_libraries(cannon ${MPI_LIBRARIES} CUDA::cudart -lcublas ${EXTERNAL_LIBS})

include_directories(".")
if(ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
endif()
