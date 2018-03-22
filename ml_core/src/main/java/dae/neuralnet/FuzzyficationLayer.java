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
    private final int batchSize;
    private String name = "fuzzy" + (count++);

    private final fmatrix inputs;
    private final imatrix bWeights;
    private final imatrix aWeights;

    private final fmatrix bWeightDeltas;
    private final fmatrix aWeightDeltas;
    private final fmatrix aWeightDeltasBatch;

    private final fmatrix sublayerOutputs;
    private final fmatrix deltas;
    private final fmatrix summedDeltas;

    private final fmatrix outputs;
    private final fmatrix errors;

    // Adam support
    // todo : split out into design
    private final float beta1 = 0.9f;
    private float beta1Corr = beta1;
    private final float beta2 = 0.999f;
    private float beta2Corr = beta2;
    private final float epsilon = 1e-8f;

    private fmatrix aMoment;
    private fmatrix aVelocity;

    private fmatrix bMoment;
    private fmatrix bVelocity;

    private ActivationFunction function;

    public FuzzyficationLayer(int nrOfInputs, int classes, int batchSize) {
        this(nrOfInputs, classes, batchSize, new fmatrix(nrOfInputs * (classes - 1), 1), new fmatrix(nrOfInputs * (classes - 1), 1), ActivationFunction.SIGMOID);
    }
    
    public FuzzyficationLayer(int nrOfInputs, int classes, int batchSize, ActivationFunction function) {
        this(nrOfInputs, classes, batchSize, new fmatrix(nrOfInputs * (classes - 1), 1), new fmatrix(nrOfInputs * (classes - 1), 1), function);
    }

    public FuzzyficationLayer(int nrOfInputs, int classes, int batchSize, imatrix weightA, imatrix weightB, ActivationFunction function) {
        this.nrOfInputs = nrOfInputs;
        this.nrOfClasses = classes;
        this.nrOfOutputs = nrOfInputs * classes;

        this.inputs = new fmatrix(nrOfInputs, 1, 1, batchSize);

        this.aWeights = weightA;
        this.aWeightDeltas = new fmatrix(nrOfInputs * (classes - 1), 1);
        this.aWeightDeltasBatch = new fmatrix(nrOfInputs * (classes - 1), 1, 1, batchSize);
        this.aMoment = new fmatrix(nrOfInputs * (classes - 1), 1);
        this.aVelocity = new fmatrix(nrOfInputs * (classes - 1), 1);

        this.bWeights = weightB;
        this.bWeightDeltas = new fmatrix(nrOfInputs * (classes - 1), 1);
        this.bMoment = new fmatrix(nrOfInputs * (classes - 1), 1);
        this.bVelocity = new fmatrix(nrOfInputs * (classes - 1), 1);

        this.sublayerOutputs = new fmatrix(nrOfInputs * (classes - 1), 1, 1, batchSize);
        this.deltas = new fmatrix(nrOfInputs * (classes - 1), 1, 1, batchSize);
        this.summedDeltas = new fmatrix(nrOfInputs * (classes - 1), 1, 1, batchSize, 1);

        this.outputs = new fmatrix(nrOfOutputs, 1, 1, batchSize);
        this.errors = new fmatrix(nrOfOutputs, 1, 1, batchSize);

        this.function = function;
        this.batchSize = batchSize;
    }

    /**
     * Returns the activation function for this layer.
     *
     * @return the activation function.
     */
    @Override
    public ActivationFunction getActivationFunction() {
        return function;
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
    
    public int getBatchSize() {
        return batchSize;
    }

    @Override
    public void forward() {
        fmatrix.fuzzyFunction(this.inputs, this.nrOfClasses, this.aWeights, this.bWeights, this.sublayerOutputs);
        fmatrix.applyActivation(function, sublayerOutputs);
        fmatrix.fuzzyShiftMinus(this.sublayerOutputs, this.nrOfClasses, this.outputs);
    }

    @Override
    public void setInputs(imatrix input) {
        fmatrix.copyInto(input, this.inputs);
    }

    @Override
    public void setIdeal(imatrix ideals) {

    }

    @Override
    public fmatrix getErrors() {
        return errors;
    }

    @Override
    public fmatrix getOutputs() {
        return outputs;
    }

    @Override
    public void backpropagate(float learningRate) {
        // errors are set.
        fmatrix.applyDerivedActivation(function, sublayerOutputs);
        fmatrix.fuzzyShiftDeltas(errors, this.nrOfClasses, deltas);
        fmatrix.dotmultiply(deltas, deltas, sublayerOutputs);

        calculateNewWeights(learningRate);
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        fmatrix.sumPerRow(deltas, summedDeltas);
        fmatrix.dotmultiply(bWeightDeltas, summedDeltas, aWeights);

        fmatrix.fuzzyInputAdd(this.inputs, bWeights, this.nrOfClasses, aWeightDeltas);

        fmatrix.dotmultiply(aWeightDeltasBatch, deltas, aWeightDeltasBatch);
        fmatrix.sumPerRow(aWeightDeltasBatch, aWeightDeltas);
    }

    @Override
    public void calculateErrors(imatrix errors) {
        fmatrix.fuzzyBackProp(deltas, aWeights, nrOfClasses, errors);
        errors.sync();
    }

    @Override
    public void adaptWeights(float factor) {
        //fmatrix.dotadd(aWeights, 1, aWeights, -factor, aWeightDeltas);
        //fmatrix.dotadd(bWeights, 1, bWeights, -factor, bWeightDeltas);
        adaptAWeights(factor);
        adaptBWeights(factor);
        beta1Corr *= 0.9f;
        beta2Corr *= 0.9f;
    }

    private void adaptAWeights(float factor) {
        fmatrix.dotadd(aMoment, beta1, aMoment, 1 - beta1, aWeightDeltas);
        fmatrix.adamVelocity(aVelocity, beta2, aVelocity, aWeightDeltas);
        fmatrix.adamAdaptWeights(aWeights, -factor, beta1Corr, beta2Corr, epsilon, aMoment, aVelocity);
    }

    private void adaptBWeights(float factor) {
        fmatrix.dotadd(bMoment, beta1, bMoment, 1 - beta1, bWeightDeltas);
        fmatrix.adamVelocity(bVelocity, beta2, bVelocity, bWeightDeltas);
        fmatrix.adamAdaptWeights(bWeights, -factor, beta1Corr, beta2Corr, epsilon, bMoment, bVelocity);

    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {
        fmatrix.randomize(aWeights, r, 0.2f, 10);
        fmatrix.randomize(bWeights, r, -10, -1);

        for (int row = 0; row < bWeights.getNrOfRows(); ++row) {
            int iClass = (nrOfClasses - (row % (nrOfClasses - 1))) - nrOfClasses / 2;
            bWeights.set(row, 0, iClass * 5 + min + r.nextFloat() * (max - min));
        }
    }

    @Override
    public void writeWeightImage(String file) {

        Path bp = Paths.get(file + "_b");
        fmatrix.writeAs2DImage(bWeights, nrOfInputs, nrOfClasses - 1, bp);

        Path ap = Paths.get(file + "_a");
        fmatrix.writeAs2DImage(aWeights, nrOfInputs, nrOfClasses - 1, ap);
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

    /**
     * Syncs the matrices with the matrices on the gpu.
     */
    @Override
    public void sync() {
        aWeights.sync();
        bWeights.sync();
    }

    
}
