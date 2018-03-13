/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.cost;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class QuadraticCostFunction implements CostFunction {

    /**
     * Calculates the loss matrix, given the output of the neural network x and
     * the desired output y.
     *
     * @param loss the loss matrix to calculate.
     * @param x the output x of the neural network.
     * @param y the desired output of the neural network.
     */
    @Override
    public void calculateCost(imatrix loss, imatrix x, imatrix y) {

        fmatrix.copyInto(x, loss);
        fmatrix.dotsubtract(loss, loss, y);
        loss.applyFunction(xv -> xv * xv);
    }

    /**
     * Calculates the derived cost per output.
     *
     * @param deltas the deltas to backpropagate in the network.
     * @param x the output x of the neural network.
     * @param y the desired output of the neural network.
     */
    @Override
    public void calculateDerivedCost(imatrix deltas, imatrix x, imatrix y) {
        fmatrix.dotsubtract(deltas, x, y);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof QuadraticCostFunction;
    }
}
