#include "densecrf.h"
#include <iostream>

int main(int argc, char const *argv[])
{
    // https://www.cs.ubc.ca/~schmidtm/Software/UGM/small.html
    constexpr size_t nStates = 2;                                               // {false, true}
    constexpr size_t nNodes  = 4;                                               // four students

    Eigen::Matrix<float, nStates, nNodes> unary;
    //       Cathy, Heather, Mark, Allison
    unary << 0.75, 0.10, 0.75, 0.10,    // wrong
             0.25, 0.90, 0.25, 0.90;    // right

    // unary << 0.25, 0.90, 0.25, 0.90,    // right
    //          0.75, 0.10, 0.75, 0.10;    // wrong

    std::cout << "unary:" << std::endl << unary << std::endl;

    DenseCRF crf(nNodes, nStates);

    crf.setUnaryEnergy( unary );

    Eigen::MatrixXf feature( 2, nNodes );
    feature << 1, 0, 1, 0,
               0, 1, 0, 1;
    crf.addPairwiseEnergy(feature, new PottsCompatibility( 2 ));

    Eigen::Matrix<short,Eigen::Dynamic,1> map = crf.map(10);
    std::cout << "map:" << std::endl << map.transpose() << std::endl;

    // Eigen::MatrixXf probs = crf.inference(10);
    // std::cout << "probs:" << std::endl << probs << std::endl;

	// Eigen::MatrixXf Q = crf.startInference(), t1, t2;
	// printf("kl = %f\n", crf.klDivergence(Q) );
	// for( int it=0; it<5; it++ ) {
	// 	crf.stepInference( Q, t1, t2 );
	// 	printf("kl = %f\n", crf.klDivergence(Q) );
	// }

	// Eigen::MatrixXf Q( nStates, nNodes ), tmp1, tmp2;
	// const PairwisePotential*pp = crf.getPotential(0);
	// crf.expAndNormalize( Q, -unary );
	// std::cout << "probs " << 'X' << ":" << std::endl << Q << std::endl;
	// for( int it=0; it<10; it++ ) {
	// 	tmp1 = -unary;
	// 	for( unsigned int k=0; k<1; k++ ) {
	// 		pp->apply( tmp2, Q );
	// 		tmp1 -= tmp2;
	// 	}
	// 	crf.expAndNormalize( Q, tmp1 );
	// 	std::cout << "probs " << it << ":" << std::endl << Q << std::endl;
	// }

    return 0;
}
