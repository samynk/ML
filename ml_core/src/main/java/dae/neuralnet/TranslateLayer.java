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
        deltaWeights.multiply(-learningRate / getBatchSize());
        fmatrix.dotsubtract(newWeights, weights, deltaWeights);
    }

    @Override
    public void calculateErrors(fmatrix errors) {
        fmatrix.copyInto(this.deltas, errors);
    }

    @Override
    public void adaptWeights() {
        fmatrix.copyInto(newWeights, weights);
    }

    @Override
    public void randomizeWeights() {
        Random r = new Random();
        weights.applyFunction(x -> ((r.nextFloat() * 2) - 1.0f) / 10.0f);
    }

    @Override
    public void writeWeightImage(String file) {
        // todo implement.
    }
}
