/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.gradient;

import dae.matrix.imatrix;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public interface GradientAlgorithm {
    /**
     * Adapts the weights in the weight matrix according to the gradient.
     * @param gradientMatrix the current gradient matrix.
     * @param learningRate the current learning rate.
     */
    public void adaptWeights(imatrix gradientMatrix, float learningRate);
}
