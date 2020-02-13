include_directories( ../external/liblbfgs/include ../external/ )

find_package (Eigen3 REQUIRED NO_MODULE)

add_library( densecrf SHARED util.cpp permutohedral.cpp unary.cpp pairwise.cpp objective.cpp densecrf.cpp labelcompatibility.cpp )
target_link_libraries( densecrf PUBLIC Eigen3::Eigen )

add_library( optimization SHARED optimization.cpp )
target_link_libraries( optimization PUBLIC Eigen3::Eigen PRIVATE lbfgs )
