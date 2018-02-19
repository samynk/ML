package dae.neuralnet.matrix;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class DefaultMatrixFactory implements MatrixFactory{

    @Override
    public imatrix create(int rows, int nrOfBiases, int columns) {
        return new fmatrix(rows+nrOfBiases, columns);
    }
}
