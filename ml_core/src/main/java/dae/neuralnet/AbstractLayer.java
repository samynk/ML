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

    private final int nrOfInputs;
    private final int nrOfOutputs;
    // multiplication of the output in slices.
    private final int nrOfOutputSlices;
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
        this.nrOfOutputSlices = 1;
        this.nrOfBiases = nrOfBiases;
        this.batchSize = batchSize;

        n = nrOfInputs + nrOfBiases;
        this.inputs = new fmatrix(batchSize, n);
        this.tinputs = new tmatrix(this.inputs);

        this.deltas = new fmatrix(batchSize, nrOfOutputs);
        this.derivatives = new fmatrix(batchSize, nrOfOutputs);

        this.outputs = new fmatrix(batchSize, nrOfOutputs, nrOfOutputSlices);
        this.errors = new fmatrix(batchSize, nrOfOutputs, nrOfOutputSlices);
        this.ideal = new fmatrix(batchSize, nrOfOutputs, nrOfOutputSlices);

        function = af;
        activation = af.getActivation();
        derivedActivation = af.getDerivedActivation();
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
    public fmatrix getOutputs() {
        return this.outputs;
    }

    /**
     * Gets the squared error of this layer.
     *
     * @return the error matrix of this layer, with the result expressed as a 1
     * x nrOfOutputs matrix.
     */
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
     * Sets the inputs of this neural net. The biases will be set to one to
     * maintain their integrity.
     *
     * @param inputs the inputs to set.
     */
    public void setInputs(float... inputs) {
        this.inputs.setRow(0, inputs);
        // set biases to one.
        for (int i = nrOfInputs; i < n; i++) {
            this.inputs.set(0, i, 1);
        }
    }

    /**
     * Copies the inputs matrix into the input matrix of this layer. The bias
     * terms will be reset to one after this operation.
     *
     * @param inputs the inputs to copy.
     */
    public void setInputs(imatrix inputs) {
        fmatrix.copyInto(inputs, this.inputs);
        for (int i = nrOfInputs; i < n; i++) {
            for (int r = 0; r < inputs.getNrOfRows(); ++r) {
                this.inputs.set(r, i, 1);
            }
        }
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
    public void setIdeal(imatrix ideals) {
        fmatrix.copyInto(ideals, this.ideal);
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
     * @param calculateErrors calculate the deltas by subtracting the ideals
     * from the outputs.
     */
    @Override
    public void backpropagate(float learningRate, boolean calculateErrors) {

        // 2. multiply with derivative of activation function. 
        // Note: derivative is f'(net input) but this is typically expressed
        // in terms of the output of the activation function.
        // 2.a copy the outputs into the derivatives matrix.
        fmatrix.copyInto(this.outputs, this.derivatives);
        // 2.b apply the derivative of the activation function to the output.
        derivatives.applyFunction(this.derivedActivation);

        // 1. copy output - ideal (target) into deltas.
        if (calculateErrors) {
            fmatrix.dotsubtract(errors, this.outputs, this.ideal);
        }

        // 3. multiply the derivatives with the errors.
        fmatrix.dotmultiply(deltas, errors, derivatives);

        // 4. Calculate the new weights
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
    public abstract void calculateErrors(fmatrix errors);

    /**
     * Apply the changes in weights to the weight matrix.
     */
    @Override
    public abstract void adaptWeights();

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
