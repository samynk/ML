package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.tmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.activation.Function;
import java.util.Random;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public abstract class AbstractLayer implements ILayer {

    private String name;

    private final int nrOfInputs;
    private final int nrOfOutputs;
    private final int nrOfBiases;
    private final int batchSize;

    private final int n;

    // inputs
    protected final fmatrix inputs;
    protected final imatrix tinputs;

    // outputs
    protected final fmatrix outputs;

    protected final fmatrix deltas;
    protected final fmatrix derivatives;

    protected final fmatrix errors;

    // target of this layer.
    protected final fmatrix ideal;

    // activation function
    protected final ActivationFunction function;
    protected final Function activation;
    protected final Function derivedActivation;

    /**
     * Creates a new AbstractLayer with a batch size of 1.
     *
     * @param nrOfInputs the number of inputs for the layer.
     * @param nrOfBiases the number of biases in the layer.
     * @param nrOfOutputs the number of outputs of the layer.
     * @param af the activation function for the outputs.
     */
    public AbstractLayer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, ActivationFunction af) {
        this(nrOfInputs, nrOfBiases, nrOfOutputs, 1, af);
    }

    /**
     * Creates a new AbstractLayer with a batch size of 1.
     *
     * @param nrOfInputs the number of inputs for the layer.
     * @param nrOfBiases the number of biases in the layer.
     * @param nrOfOutputs the number of outputs of the layer.
     * @param batchSize the batch size for the layer.
     * @param af the activation function for the outputs.
     */
    public AbstractLayer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, int batchSize, ActivationFunction af) {
        this.nrOfInputs = nrOfInputs;
        this.nrOfOutputs = nrOfOutputs;
        this.nrOfBiases = nrOfBiases;
        this.batchSize = batchSize;

        n = nrOfInputs + nrOfBiases;
        this.inputs = new fmatrix(n, 1, 1, batchSize);
        // set biases to one.
        for (int h = 0; h < batchSize; ++h) {
            for (int i = nrOfInputs; i < n; i++) {
                this.inputs.set(i, 0, 0, h, 1);
            }
        }
        this.tinputs = new tmatrix(this.inputs);

        this.deltas = new fmatrix(nrOfOutputs, 1, 1, batchSize);
        this.derivatives = new fmatrix(nrOfOutputs, 1, 1, batchSize);

        this.outputs = new fmatrix(nrOfOutputs, 1, 1, batchSize);
        this.errors = new fmatrix(nrOfOutputs, 1, 1, batchSize);
        this.ideal = new fmatrix(nrOfOutputs, 1, 1, batchSize);

        function = af;
        activation = af.getActivation();
        derivedActivation = af.getDerivedActivation();
    }

    /**
     * Returns the activation function for this layer.
     *
     * @return the activation function.
     */
    @Override
    public ActivationFunction getActivationFunction() {
        return this.function;
    }

    /**
     * Set the name of this layer.
     *
     * @param name the name of the layer.
     */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the name of this layer.
     *
     * @return the name of the layer.
     */
    @Override
    public String getName() {
        return name;
    }

    /**
     * @return the nrOfInputs
     */
    @Override
    public int getNrOfInputs() {
        return nrOfInputs;
    }

    /**
     * @return the nrOfBiases
     */
    public int getNrOfBiases() {
        return nrOfBiases;
    }

    /**
     * @return the nrOfOutputs
     */
    @Override
    public int getNrOfOutputs() {
        return nrOfOutputs;
    }

    /**
     * Return the number of batches.
     *
     * @return the number of batches.
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Returns the matrix with the outputs of this neural network layer, with
     * the application of the activation function.
     *
     * @return the output matrix of this layer, with the result expressed as a 1
     * x nrOfOutputs matrix.
     */
    @Override
    public fmatrix getOutputs() {
        return this.outputs;
    }

    /**
     * Gets the error vector of this layer.
     *
     * @return the error matrix of this layer, with the result expressed as a 1
     * x nrOfOutputs matrix.
     */
    @Override
    public fmatrix getErrors() {
        return errors;
    }

    /**
     * Returns the ideals of this layer.
     *
     * @return the matrix with the ideal outputs for this layer.
     */
    public fmatrix getDeltas() {
        return this.deltas;
    }

    /**
     * Copies the inputs matrix into the input matrix of this layer. The bias
     * terms will be reset to one after this operation.
     *
     * @param inputs the inputs to copy.
     */
    @Override
    public void setInputs(imatrix inputs) {
        fmatrix.copyIntoSlice(inputs, this.inputs);
    }
    
    @Override
    public imatrix getInputs() {
        return inputs;
    }

    /**
     * Sets the ideal values for this layer.
     *
     * @param ideal the ideal values.
     */
    public void setIdeal(float... ideal) {
        this.ideal.setRow(0, ideal);
    }

    /**
     * Sets the ideal values for this layer.
     *
     * @param ideals the ideal values.
     */
    @Override
    public void setIdeal(imatrix ideals) {
        fmatrix.copyInto(ideals, this.ideal);
    }

    /**
     * Returns the target matrix.
     *
     * @return the target of the neural network.
     */
    public imatrix getIdeal() {
        return this.ideal;
    }

    /**
     * Multiplies the input matrix with the weight matrix and stores the result
     * in the output matrix.
     */
    @Override
    public abstract void forward();

    /**
     * Performs the back propagation algorithm with the provided ideals and the
     * learningRate. The new weights must be calculated but not yet applied to
     * the actual weights.
     *
     * @param learningRate the learning rate for the back propagation.
     */
    @Override
    public void backpropagate(float learningRate) {

        // 1. multiply with derivative of activation function. 
        // Note: derivative is f'(net input) but this is typically expressed
        // in terms of the output of the activation function.
        // 1.a copy the outputs into the derivatives matrix.
        fmatrix.copyInto(this.outputs, this.derivatives);
        // 1.b apply the derivative of the activation function to the output.
        fmatrix.applyDerivedActivation(this.function, derivatives);

        // 2. multiply the derivatives with the errors.
        fmatrix.dotmultiply(deltas, errors, derivatives);
        // 3. Calculate the new weights
        calculateNewWeights(learningRate);
    }

    /**
     * Calculates the new weights for the system given the learning rate. This
     * method is called from the backpropagate function to ensure that the
     * correct deltas were calculated.
     *
     * @param learningRate
     */
    @Override
    public abstract void calculateNewWeights(float learningRate);

    /**
     * Calculate the deltas for this layer, given the deltas of the next layer.
     *
     * @param errors the deltas of the next layer.
     */
    @Override
    public abstract void calculateErrors(imatrix errors);

    /**
     * Apply the changes in weights to the weight matrix.
     *
     * @param factor the factor for the weights.
     */
    @Override
    public abstract void adaptWeights(float factor);

    /**
     * Randomize all the weights.
     *
     * @param r the random object to use.
     * @param min the minimum value for the random weight.
     * @param max the maximum value for the random weight.
     */
    @Override
    public abstract void randomizeWeights(Random r, float min, float max);

    @Override
    public abstract void writeWeightImage(String file);

}
