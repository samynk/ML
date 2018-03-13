/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.analysis.WeightAnalysis;
import dae.neuralnet.analysis.WeightAnalyzer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FuzzyficationLayer implements ILayer {

    private final int nrOfInputs;
    private final int nrOfClasses;
    private final int nrOfOutputs;

    private static int count = 0;
    private String name = "fuzzy" + (count++);

    private final fmatrix inputs;
    private final imatrix bWeights;
    private final imatrix aWeights;

    private final fmatrix bWeightDeltas;
    private final fmatrix aWeightDeltas;

    private final fmatrix bBatchWeights;
    private final fmatrix aBatchWeights;

    private final fmatrix sublayerOutputs;
    private final fmatrix deltas;

    private final fmatrix outputs;
    private final fmatrix errors;
    private final fmatrix errorVector;

    public FuzzyficationLayer(int nrOfInputs, int classes) {
        this(nrOfInputs, classes, new fmatrix(nrOfInputs, classes - 1), new fmatrix(nrOfInputs, classes - 1));
    }

    public FuzzyficationLayer(int nrOfInputs, int classes, imatrix weightA, imatrix weightB) {
        this.nrOfInputs = nrOfInputs;
        this.nrOfClasses = classes;
        this.nrOfOutputs = nrOfInputs * classes;

        this.inputs = new fmatrix(1, nrOfInputs);

        this.aWeights = weightA;
        this.aWeightDeltas = new fmatrix(nrOfInputs, classes - 1);
        this.aBatchWeights = new fmatrix(nrOfInputs, classes - 1);

        this.bWeights = weightB;
        this.bWeightDeltas = new fmatrix(nrOfInputs, classes - 1);
        this.bBatchWeights = new fmatrix(nrOfInputs, classes - 1);

        this.sublayerOutputs = new fmatrix(nrOfInputs, classes - 1);
        this.deltas = new fmatrix(nrOfInputs, classes - 1);

        this.outputs = new fmatrix(nrOfInputs, classes);
        this.errors = new fmatrix(nrOfInputs, classes);
        this.errorVector = new fmatrix(1, errors.getSize());
    }

    /**
     * Returns the activation function for this layer.
     *
     * @return the activation function.
     */
    @Override
    public ActivationFunction getActivationFunction() {
        return ActivationFunction.SIGMOID;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public int getNrOfInputs() {
        return nrOfInputs;
    }

    @Override
    public int getNrOfOutputs() {
        return nrOfOutputs;
    }

    public imatrix getBWeights() {
        return bWeights;
    }

    public imatrix getAWeights() {
        return aWeights;
    }

    public int getNrOfClasses() {
        return nrOfClasses;
    }

    @Override
    public void forward() {
        fmatrix.fuzzyFunction(this.inputs, this.aWeights, this.bWeights, this.sublayerOutputs);
        fmatrix.fuzzyShiftMinus(this.sublayerOutputs, this.outputs);
    }

    @Override
    public void setInputs(imatrix input) {
        if (input.isRowVector()) {
            fmatrix.copyInto(input, this.inputs);
        } else {
            fmatrix.matrixToRowVector(input, this.inputs);
        }
    }

    @Override
    public void setIdeal(imatrix ideals) {

    }

    @Override
    public fmatrix getErrors() {
        return errorVector;
    }

    @Override
    public fmatrix getOutputs() {
        return outputs;
    }

    @Override
    public void backpropagate(float learningRate) {
        // errors are set.
        fmatrix.dsigmoid(sublayerOutputs);
        fmatrix.rowVectorToMatrix(errorVector, errors);
        fmatrix.fuzzyShiftDeltas(errors, deltas);
        fmatrix.dotmultiply(deltas, deltas, sublayerOutputs);

        calculateNewWeights(learningRate);

    }

    @Override
    public void calculateNewWeights(float learningRate) {
        fmatrix.dotmultiply(bWeightDeltas, deltas, aWeights);
        fmatrix.dotadd(this.bBatchWeights, bBatchWeights, bWeightDeltas);

        fmatrix.copyInto(bWeights, aWeightDeltas);
        for (int r = 0; r < aWeightDeltas.getNrOfRows(); ++r) {
            float iv = inputs.get(0, r);
            for (int c = 0; c < aWeightDeltas.getNrOfColumns(); ++c) {
                float current = aWeightDeltas.get(r, c);
                aWeightDeltas.set(r, c, current + iv);
            }
        }
        fmatrix.dotmultiply(aWeightDeltas, deltas, aWeightDeltas);
        fmatrix.dotadd(this.aBatchWeights, aBatchWeights, aWeightDeltas);
    }

    @Override
    public void calculateErrors(fmatrix errors) {
        fmatrix.sumPerRow(bWeightDeltas, errors);
    }

    @Override
    public void adaptWeights(float factor) {

        fmatrix.dotadd(aWeights, 1, aWeights, -factor, aBatchWeights);
        fmatrix.dotadd(bWeights, 1, bWeights, -factor, bBatchWeights);

        //aWeights.applyFunction(x -> x > .001f ? x : 0.001f);
        aBatchWeights.reset();
        bBatchWeights.reset();
    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {
        fmatrix.randomize(aWeights, r, min, max);
        fmatrix.randomize(bWeights, r, min, max);
    }

    @Override
    public void writeWeightImage(String file) {

        Path bp = Paths.get(file + "_b");
        fmatrix.writeAs2DImage(bWeights, bp);

        Path ap = Paths.get(file + "_a");
        fmatrix.writeAs2DImage(aWeights, ap);
    }

    @Override
    public void writeOutputImage(String file) {

    }

    @Override
    public void analyzeWeights() {
        WeightAnalysis wa1 = WeightAnalyzer.analyzeMatrix(this.aWeights);
        System.out.println("weight analysis of " + this.name + " a weights");
        System.out.println(wa1);

        WeightAnalysis wa2 = WeightAnalyzer.analyzeMatrix(this.bWeights);
        System.out.println("weight analysis of " + this.name + " b weights");
        System.out.println(wa2);
    }
}
