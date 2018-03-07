/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.cost;

import dae.matrix.imatrix;

/**
 * An interface
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public interface CostFunction {

    /**
     * Calculates the loss matrix, given the output of the neural network x and
     * the desired output y.
     *
     * @param loss the loss matrix to calculate.
     * @param x the output x of the neural network.
     * @param y the desired output of the neural network.
     */
    public void calculateCost(imatrix loss, imatrix x, imatrix y);

    /**
     * Calculates the derived cost per output.
     *
     * @param deltas the deltas to backpropagate in the network.
     * @param x the output x of the neural network.
     * @param y the desired output of the neural network.
     */
    public void calculateDerivedCost(imatrix deltas, imatrix x, imatrix y);
}
