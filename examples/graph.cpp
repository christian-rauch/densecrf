#include "densecrf.h"
#include <iostream>
#include <iomanip>

int main(int argc, char const *argv[])
{
    /*
     *  0 0 X X
     *  0 0 X 0
     *  0 0 X X
     */

    constexpr size_t rows = 3, cols = 4;
    constexpr size_t npoints = rows * cols;
    constexpr size_t nstates = 2;

    Eigen::Matrix<float, nstates, npoints> unary;
    unary << // probability of 'X'
             0, 0, 1, 1,
             0, 0, 1, 0,
             0, 0, 1, 1,
             // probability of not 'X'
             1, 1, 0, 0,
             1, 1, 0, 1,
             1, 1, 0, 0;
    DenseCRF crf(npoints, nstates);
    crf.setUnaryEnergy( unary );

    std::cout << "unary:" << std::endl << unary << std::endl;

    // feature: 2D location
    Eigen::MatrixXf feature( 2, npoints );
    feature << // x coordinate
               0, 1, 2, 3,
               0, 1, 2, 3,
               0, 1, 2, 3,
               // y coordinate
               0, 0, 0, 0,
               1, 1, 1, 1,
               2, 2, 2, 2;
    crf.addPairwiseEnergy(feature, new PottsCompatibility());

    std::cout << "feature:" << std::endl << feature << std::endl;

    // Eigen::MatrixXf probs = crf.inference(100);
    // std::cout << "probs:" << std::setprecision(2) << std::endl << probs << std::endl;

    // const Eigen::VectorXf p = probs.row(1);
    // std::cout << "probs:" << std::setprecision(2) << std::endl << Eigen::Map<const Eigen::Matrix<float, rows, cols, Eigen::RowMajor>>(p.data()) << std::endl;

	Eigen::MatrixXf Q( nstates, npoints ), tmp1, tmp2;
	const PairwisePotential*pp = crf.getPotential(0);
	crf.expAndNormalize( Q, -unary );
	std::cout << "probs " << 'X' << ":" << std::endl << Q << std::endl;
	for( int it=0; it<10; it++ ) {
		tmp1 = -unary;
		for( unsigned int k=0; k<1; k++ ) {
			pp->apply( tmp2, Q );
			tmp1 -= tmp2;
		}
		crf.expAndNormalize( Q, tmp1 );
		std::cout << "probs " << it << ":" << std::endl << Q << std::endl;
	}

    return 0;
}
