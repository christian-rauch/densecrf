include_directories( ../external/liblbfgs/include ../external/ )

find_package (Eigen3 REQUIRED NO_MODULE)

add_library( densecrf util.cpp permutohedral.cpp unary.cpp pairwise.cpp objective.cpp densecrf.cpp labelcompatibility.cpp )
target_link_libraries( densecrf Eigen3::Eigen )

add_library( optimization optimization.cpp )
target_link_libraries( optimization lbfgs Eigen3::Eigen )