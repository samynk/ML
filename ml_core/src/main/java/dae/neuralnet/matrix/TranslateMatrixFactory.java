package dae.neuralnet.matrix;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class TranslateMatrixFactory implements MatrixFactory{

    @Override
    public imatrix create(int inputs, int nrOfBiases, int outputs) {
        if ( inputs != outputs ){
            throw new IllegalArgumentException("Number of inputs should be equal to number of outputs.");
        }
        if ( nrOfBiases != 1){
            throw new IllegalArgumentException("Number of biases should be one.");
        }
        
        return new fmatrix(inputs+nrOfBiases, outputs);
    }
    
}
