add_executable(test_cannon test_cannon.cpp)

target_link_libraries(test_cannon cannon gtest pthread)

add_test(CannonTest mpirun -n 4 ./test_cannon 100)
