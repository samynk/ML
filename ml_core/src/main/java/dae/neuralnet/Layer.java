package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.tmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.analysis.WeightAnalysis;
import dae.neuralnet.analysis.WeightAnalyzer;
import dae.neuralnet.gradient.AdamGradientAlgorithm;
import dae.neuralnet.gradient.GradientAlgorithm;
import java.nio.file.Paths;
import java.util.Random;

/**
 * The neural network layer is the basic interface that defines the
 * functionality of a neural network layer. A layer has a fixed number of
 * variable inputs, bias nodes and output nodes.
 *
 * The weights of the layer can be a matrix that has the dimension (inputs+bias)
 * x (outputs), but other configurations are also allowed (as for example in
 * convolutional neural networks).
 *
 * @author Koen Samyn
 */
public class Layer extends AbstractLayer {

    // adam test
    /**
     * The gradient algorithm.
     */
    private GradientAlgorithm gradientAlgorithm;

    private final imatrix weights;
    private final imatrix tweights;
    private final imatrix tdeltas;

    private imatrix constraint;

    private final imatrix deltaWeights;

    private float dropRate = 0;
    private boolean dropRateSet = false;
    private Random dropRandom;
    private imatrix dropWeightMatrix;
    private imatrix tDropWeightMatrix;

    /**
     * Creates a layer with a given number of inputs, biases and outputs.
     *
     * The number of inputs n is defined as: nrOfInputs+nrOfBiases.
     *
     * The input matrix is then defined as an (1 x n) matrix. The bias values
     * will be set to one.
     *
     * The number of outputs m is equal to the nrOfOutputs parameter.
     *
     * The weight matrix is thus defined as an (n x m) matrix.
     *
     * Multiplication of the (1xn) matrix with the (n x m) matrix leads to an
     * output matrix with dimension (1 x m).
     *
     * @param nrOfInputs the number of inputs of this layer.
     * @param nrOfBiases the number of biases of this layer.
     * @param nrOfOutputs the number of outputs of this layer.
     * @param af the activation function
     */
    public Layer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, ActivationFunction af) {
        this(nrOfInputs, nrOfBiases, nrOfOutputs, 1, af);
    }

    /**
     * Creates a layer with a given number of inputs, biases and outputs.
     *
     * The number of inputs n is defined as: nrOfInputs+nrOfBiases.
     *
     * The input matrix is then defined as an (batchSize x n) matrix. The bias
     * values will be set to one.
     *
     * The number of outputs m is equal to the nrOfOutputs parameter.
     *
     * The weight matrix is thus defined as an (n x m) matrix.
     *
     * Multiplication of the (batchSizexn) matrix with the (n x m) matrix leads
     * to an output matrix with dimension (batchSize x m).
     *
     * @param nrOfInputs the number of inputs of this layer.
     * @param nrOfBiases the number of biases of this layer.
     * @param nrOfOutputs the number of outputs of this layer.
     * @param batchSize the batch size
     * @param af the activation function
     */
    public Layer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, int batchSize, ActivationFunction af) {
        this(nrOfInputs, nrOfBiases, nrOfOutputs, batchSize, af, new fmatrix(nrOfInputs + nrOfBiases, nrOfOutputs));
    }

    /**
     * Creates a layer with a given number of inputs, biases and outputs.
     *
     * The number of inputs n is defined as: nrOfInputs+nrOfBiases.
     *
     * The input matrix is then defined as an (batchSize x n) matrix. The bias
     * values will be set to one.
     *
     * The number of outputs m is equal to the nrOfOutputs parameter.
     *
     * The weight matrix is thus defined as an (n x m) matrix.
     *
     * Multiplication of the (batchSizexn) matrix with the (n x m) matrix leads
     * to an output matrix with dimension (batchSize x m).
     *
     * @param nrOfInputs the number of inputs of this layer.
     * @param nrOfBiases the number of biases of this layer.
     * @param nrOfOutputs the number of outputs of this layer.
     * @param batchSize the batch size
     * @param af the activation function
     * @param weights the weights to initialize the Layer with.
     */
    public Layer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, int batchSize, ActivationFunction af, imatrix weights) {
        super(nrOfInputs, nrOfBiases, nrOfOutputs, batchSize, af);
        this.weights = weights;
        
        gradientAlgorithm = new AdamGradientAlgorithm(weights);

        this.tweights = new tmatrix(weights);
        this.tdeltas = new tmatrix(deltas);
        this.deltaWeights = new fmatrix(nrOfInputs + nrOfBiases, nrOfOutputs);
    }

    /**
     * Sets the dropRate for the weights in this layer.
     *
     * @param dropRate the drop rate for the weights
     */
    public void setDropRate(float dropRate) {
        this.dropRate = dropRate;
        this.dropRateSet = true;
        this.dropRandom = new Random(System.currentTimeMillis());
        this.constraint = new fmatrix(getNrOfInputs() + getNrOfBiases(), getNrOfOutputs());
        this.dropWeightMatrix = new fmatrix(getNrOfInputs() + getNrOfBiases(), getNrOfOutputs());
        this.tDropWeightMatrix = new tmatrix(dropWeightMatrix);
    }

    public boolean isDropRateSet() {
        return dropRateSet;
    }

    public float getDropRate() {
        return dropRate;
    }

    /**
     * A matrix that constraints the weights that can be changed.
     *
     * @param constraint the constraint matrix.
     */
    public void setConstraint(imatrix constraint) {
        this.constraint = constraint;
    }

    /**
     * Returns the matrix with the weights of the single neural network layer.
     *
     * @return the weights of the neural net.
     */
    public imatrix getWeights() {
        return this.weights;
    }

    /**
     * Returns the deltas for the weights in this layer.
     *
     * @return the matrix with the delta weights.
     */
    public imatrix getDeltaWeights() {
        return this.deltaWeights;
    }

    /**
     * Multiplies the input matrix with the weight matrix and stores the result
     * in the output matrix.
     */
    @Override
    public void forward() {
        //outputs.reset();
        imatrix weightMatrix = tweights;
        if (dropRateSet) {
            this.constraint.applyFunction(x -> dropRandom.nextFloat() > dropRate ? 1 : 0);
            fmatrix.dotmultiply(dropWeightMatrix, constraint, weights);
            weightMatrix = tDropWeightMatrix;
        }
        fmatrix.sgemm(1, weightMatrix, inputs, 0, outputs);
        //fmatrix.sgemm(1, inputs, weightMatrix, 0, outputs);

        switch (function) {
            case SOFTMAX:
                outputs.sync();
                fmatrix.softMaxPerRow(outputs);
                outputs.makeMaster();
                break;

            default:
                fmatrix.applyActivation(this.function, outputs);
        }
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        fmatrix.sgemm(-1, inputs, tdeltas, 0, deltaWeights);
        if (dropRateSet) {
            fmatrix.dotmultiply(deltaWeights, deltaWeights, constraint);
        }
    }

    /**
     * Calculates the errors that can be used in a previous layer.
     *
     * @param errors the error matrix with dimension (batchSize x nrOfInputs).
     */
    @Override
    public void calculateErrors(imatrix errors) {
        // errors.reset();
        imatrix t = weights;
        if (dropRateSet) {
            t = dropWeightMatrix;
        }
        fmatrix.sgemm(1, t, deltas, 0, errors);
        //fmatrix.multiply(deltas, this.deltas, this.tweights);
    }

    @Override
    public void adaptWeights(float factor) {
        gradientAlgorithm.adaptWeights(deltaWeights, factor);
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
        weights.applyFunction(x -> (float)r.nextGaussian() / 10.0f);
    }

    public void printInputs() {
        System.out.println(inputs.toString());
    }

    @Override
    public void writeWeightImage(String file) {
        fmatrix.writeAs2DImage(weights, Paths.get(file));
    }

    @Override
    public void writeOutputImage(String file) {

    }

    @Override
    public void analyzeWeights() {
        WeightAnalysis wa1 = WeightAnalyzer.analyzeMatrix(this.weights);
        System.out.println("weight analysis of " + getName() + " weights");
        System.out.println(wa1);
    }

    /**
     * Syncs the matrices with the matrices on the gpu.
     */
    @Override
    public void sync() {
        this.weights.sync();
    }
}
