package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.neuralnet.activation.ActivationFunction;
import java.util.Random;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class TranslateLayer extends AbstractLayer {

    private final fmatrix weights;
    private final fmatrix deltaWeights;
    private final fmatrix newWeights;

    /**
     * Creates a new translation layer which applies a translation transform on
     * the input. The number of biases is automatically set to 1 and the number
     * of inputs and outputs is set to n.
     *
     * @param n the number of inputs and outputs.
     * @param batchSize the batch size.
     * @param af the activation function for this layer.
     */
    public TranslateLayer(int n, int batchSize, ActivationFunction af) {
        super(n, 1, n, batchSize, af);
        weights = new fmatrix(1, n);
        deltaWeights = new fmatrix(1, n);
        newWeights = new fmatrix(1, n);
    }

    /**
     * Creates a new translation layer which applies a translation transform on
     * the input. The number of biases is automatically set to 1 and the number
     * of inputs and outputs is set to n.
     *
     * @param n the number of inputs and outputs.
     * @param af the activation function for this layer.
     */
    public TranslateLayer(int n, ActivationFunction af) {
        super(n, 1, n, af);
        weights = new fmatrix(1, n);
        deltaWeights = new fmatrix(1, n);
        newWeights = new fmatrix(1, n);
    }

    @Override
    public void forward() {
        fmatrix.copyInto(inputs, outputs);
        for (int row = 0; row < outputs.getNrOfRows(); ++row) {
            fmatrix.dotaddrow(row, outputs, row, outputs, 0, weights);
        }
        outputs.applyFunction(activation);
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        fmatrix.sumPerColumn(errors, deltaWeights);
    }

    @Override
    public void calculateErrors(fmatrix errors) {
        fmatrix.copyInto(this.deltas, errors);
    }

    @Override
    public void adaptWeights(float factor) {
        fmatrix.copyInto(newWeights, weights);
    }

    /**
     * Randomize all the weights.
     *
     * @param r the Random object to use.
     * @param min the minimum value for the random weight.
     * @param max the maximum value for the random weight.
     */
    @Override
    public void randomizeWeights(Random r, float min, float max) {
        weights.applyFunction(x -> min + r.nextFloat() * (max - min));
    }

    @Override
    public void writeWeightImage(String file) {
        // todo implement.
    }

    @Override
    public void writeOutputImage(String file) {
    }
}
