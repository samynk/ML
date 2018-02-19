package dae.neuralnet.matrix;

import dae.matrix.imatrix;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public interface MatrixFactory {
    public static final DefaultMatrixFactory DEFAULT = new DefaultMatrixFactory();
    public static final TranslateMatrixFactory TRANSLATE_MATRIX = new TranslateMatrixFactory();
    
    
    public imatrix create( int inputs, int nrOfBiases, int outputs);
}
