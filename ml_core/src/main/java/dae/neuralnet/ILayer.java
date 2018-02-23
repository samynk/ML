package dae.neuralnet;

import dae.matrix.fmatrix;
import java.util.Random;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public interface ILayer {

    /**
     * @return the total number of inputs.
     */
    public int getNrOfInputs();

    /**
     * @return the total number of outputs.
     */
    public int getNrOfOutputs();

    /**
     * Multiplies the input matrix with the weight matrix and stores the result
     * in the output matrix.
     */
    public void forward();

    /**
     * Performs the back propagation algorithm with the provided ideals and the
     * learningRate. The new weights must be calculated but not yet applied to
     * the actual weights.
     *
     * @param learningRate the learning rate for the back propagation.
     * @param calculateErrors calculate the deltas by subtracting the ideals
     * from the outputs.
     */
    public void backpropagate(float learningRate, boolean calculateErrors);

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
    public void calculateErrors(fmatrix errors);

    /**
     * Apply the changes in weights to the weight matrix.
     */
    public void adaptWeights();

    /**
     * Randomize all the weights.
     *
     * @param r the Random object to use.
     * @param min the minimum value for the random weight.
     * @param max the maximum value for the random weight.
     */
    public void randomizeWeights(Random r, float min, float max);

    /**
     * Writes the weights as an image to the given file location.
     *
     * @param file the file location to write to.
     */
    public void writeWeightImage(String file);
}
