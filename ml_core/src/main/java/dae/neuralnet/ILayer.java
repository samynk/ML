package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
import java.util.Random;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public interface ILayer {

    /**
     * Set the name of this layer.
     *
     * @param name the name of the layer.
     */
    public void setName(String name);

    /**
     * Returns the name of this layer.
     *
     * @return the name of the layer.
     */
    public String getName();

    /**
     * @return the total number of inputs.
     */
    public int getNrOfInputs();

    /**
     * @return the total number of outputs.
     */
    public int getNrOfOutputs();
    
    /**
     * Returns the activation function for this layer.
     * @return the activation function.
     */
    public ActivationFunction getActivationFunction();

    /**
     * Multiplies the input matrix with the weight matrix and stores the result
     * in the output matrix.
     */
    public void forward();

    /**
     * Sets the inputs of this layer.
     *
     * @param input the input of this layer.
     */
    public void setInputs(imatrix input);

    /**
     * Sets the ideal values for this layer.
     *
     * @param ideals the ideal values.
     */
    public void setIdeal(imatrix ideals);

    /**
     * Gets the error vector of this layer.
     *
     * @return the error matrix of this layer, with the result expressed as a 1
     * x nrOfOutputs matrix.
     */
    public imatrix getErrors();

    /**
     * Returns the matrix with the outputs of this neural network layer, with
     * the application of the activation function.
     *
     * @return the output matrix of this layer, with the result expressed as a 1
     * x nrOfOutputs matrix.
     */
    public fmatrix getOutputs();

    /**
     * Performs the back propagation algorithm with the provided ideals and the
     * learningRate. The new weights must be calculated but not yet applied to
     * the actual weights.
     *
     * @param learningRate the learning rate for the back propagation.
     */
    public void backpropagate(float learningRate);

    /**
     * Calculates the new weights for the system given the learning rate. This
     * method is called from the backpropagate function to ensure that the
     * correct deltas were calculated.
     *
     * @param learningRate
     */
    public void calculateNewWeights(float learningRate);

    /**
     * Calculate the deltas for this layer, given the deltas of the next layer.
     *
     * @param errors the deltas of the next layer.
     */
    public void calculateErrors(imatrix errors);

    /**
     * Apply the changes in weights to the weight matrix.
     * @param factor the factor with which to multiply the weight deltas with.
     */
    public void adaptWeights(float factor);

    /**
     * Randomize all the weights.
     *
     * @param r the Random object to use.
     * @param min the minimum value for the random weight.
     * @param max the maximum value for the random weight.
     */
    public void randomizeWeights(Random r, float min, float max);
    
    /**
     * Analyzes the weights in the layer.
     *
     */
    public void analyzeWeights();

    /**
     * Writes the weights as an image to the given file location.
     *
     * @param file the file location to write to.
     */
    public void writeWeightImage(String file);

    /**
     * Writes the weights as an image to the given file location.
     *
     * @param file the file location to write to.
     */
    public void writeOutputImage(String file);
    
    /**
     * Syncs the matrices with the matrices on the gpu.
     */
    public void sync();
}
