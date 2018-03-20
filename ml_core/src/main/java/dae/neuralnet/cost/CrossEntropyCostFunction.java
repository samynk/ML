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
public class CrossEntropyCostFunction implements CostFunction {

    @Override
    public void calculateCost(imatrix loss, imatrix x, imatrix y) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void calculateDerivedCost(imatrix deltas, imatrix x, imatrix y) {
        fmatrix.dotsubtract(deltas, x, y);
//        x.sync();
//        y.sync();
//        for (int hyperslice = 0; hyperslice < y.getNrOfHyperSlices(); ++hyperslice) {
//            for (int column = 0; column < y.getNrOfColumns(); ++column) {
//                float yv = y.get(0, column, 0);
//                float xv = x.get(0, column, 0) + 0.001f;
//                float value = (xv - yv);
//                deltas.set(0, column, 0, hyperslice, value);
//            }
//        }
//        deltas.makeMaster();
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof CrossEntropyCostFunction;
    }

}
