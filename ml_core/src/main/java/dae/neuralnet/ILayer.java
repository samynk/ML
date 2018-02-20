package dae.neuralnet;

import dae.matrix.fmatrix;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public interface ILayer {
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
     */
    public void randomizeWeights();

    /**
     * Writes the weights as an image to the given file location.
     * @param file the file location to write to.
     */
    public void writeWeightImage(String file);
}
