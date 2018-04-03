/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.gradient;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class AdamGradientAlgorithm implements GradientAlgorithm {

    private final fmatrix M;
    private final fmatrix V;

    private final imatrix W;

    private final float beta1;
    private final float beta2;
    private float beta1Corr;
    private float beta2Corr;

    private final float epsilon = 1e-8f;
    private final float betaDecay = .9f;

    public AdamGradientAlgorithm(imatrix W) {
        this(W, 0.9f, 0.999f);
    }

    public AdamGradientAlgorithm(imatrix W, float beta1, float beta2) {
        this.W = W;
        M = new fmatrix(W.getNrOfRows(), W.getNrOfColumns(), W.getNrOfSlices());
        V = new fmatrix(W.getNrOfRows(), W.getNrOfColumns(), W.getNrOfSlices());

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.beta1Corr = beta1;
        this.beta2Corr = beta2;
    }

    /**
     * Adapts the weights in the weight matrix according to the gradient.
     *
     * @param g the current gradient matrix.
     * @param learningRate the current learning rate.
     */
    @Override
    public void adaptWeights(imatrix g, float learningRate) {
        fmatrix.dotadd(M, beta1, M, 1 - beta1, g);
        fmatrix.adamVelocity(V, beta2, V, g);

        // beta1 and beta2 need to die out.
        fmatrix.adamAdaptWeights(W, -learningRate, beta1Corr, beta2Corr, epsilon, M, V);
        //fmatrix.dotadd(weights, 1, weights, factor, deltaWeights);
        beta1Corr *= betaDecay;
        beta2Corr *= betaDecay;
    }

}
