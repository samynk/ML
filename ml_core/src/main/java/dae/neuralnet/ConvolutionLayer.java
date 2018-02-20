package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.fsubmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;

/**
 *
 * @author DAE_DEMO_BEAST
 */
public class ConvolutionLayer implements ILayer {

    /**
     * The number of different features we want to detect, this corresponds into
     * the number of layers.
     */
    private int features;
    /**
     * The size of the rectangular filter.
     */
    private int filterSize;
    /**
     * The stride of the convolution layer.
     */
    private int stride;

    /**
     * The weight matrices
     */
    private final fmatrix[] weights;
    /**
     * The inputs for this layer.
     */
    private final imatrix inputs;
    /**
     * The padded inputs.
     */
    private final fmatrix paddedInputs;

    /**
     * Creates a new convolution layer. The inputs of the convolutional layer
     * are interpreted as a 2D array of inputs.
     *
     * @param wInputs The number of inputs in the x direction.
     * @param hInputs The number of inputs in the y direction.
     * @param features The number of convolutional layers.
     * @param stride the stride to slide the filter with.
     * @param filter the size of filter. The total number of weights per
     * convolutional layer will be filter x filter.
     * @param padding the padding around the inputs.
     * @param batchSize the batchSize the use.
     * @param af the activation function.
     */
    public ConvolutionLayer(int features, int wInputs, int hInputs, int padding, int filter, int stride, int batchSize, ActivationFunction af) {
        // filter weights are shared.
        weights = new fmatrix[features];
        for (int i = 0; i < features; ++i) {
            weights[i] = new fmatrix(filter, filter);
        }

        paddedInputs = new fmatrix(wInputs + padding*2, hInputs + padding*2);
        inputs =new fsubmatrix(paddedInputs, padding, padding, hInputs, wInputs);
        
    }

    @Override
    public void forward() {

    }

    @Override
    public void calculateNewWeights(float learningRate) {

    }

    @Override
    public void calculateErrors(fmatrix errors) {

    }

    @Override
    public void adaptWeights() {

    }

    @Override
    public void randomizeWeights() {

    }

    @Override
    public void writeWeightImage(String file) {

    }

    @Override
    public void backpropagate(float learningRate, boolean calculateErrors) {

    }

}
