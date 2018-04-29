/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.Dimension;
import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.analysis.WeightAnalysis;
import dae.neuralnet.analysis.WeightAnalyzer;
import dae.neuralnet.gradient.AdamGradientAlgorithm;
import dae.neuralnet.gradient.GradientAlgorithm;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 * Condense corresponding cells in slices. The slices per group parameter
 * controls how much slices will be condensed into a single result.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class PancakeLayer implements ILayer {

    private final Dimension inputDimension;
    private final Dimension outputDimension;

    private String name;

    private final imatrix inputs;
    private final int slicesPerGroup;
    private final fmatrix outputs;
    private final fmatrix deltas;

    private final imatrix weights;
    private final imatrix deltaWeight;
    private final imatrix weightBatch;
    private final imatrix weightVector;
    private final boolean useBias;
    private final imatrix biases;
    private final imatrix deltaBias;
    private final imatrix biasesBatch;
    private final imatrix biasVector;

    private final GradientAlgorithm weightGradient;
    private final GradientAlgorithm biasGradient;

    private final imatrix errors;

    private final ActivationFunction function;

    /**
     * Creates a new pancake layer.
     *
     * @param inputDimension the dimensions of the input.
     * @param bias true if a bias should be added to the result.
     * @param slicesPerGroup the slicesPerGroup.
     * @param function the activation function of this pancake layer.
     */
    public PancakeLayer(Dimension inputDimension, boolean bias, int slicesPerGroup, ActivationFunction function) {
        this.inputDimension = inputDimension;
        this.outputDimension = Dimension.Dim(
                inputDimension.getRows(),
                inputDimension.getColumns(),
                inputDimension.getSlices() / slicesPerGroup,
                inputDimension.getHyperSlices());
        useBias = bias;
        this.slicesPerGroup = slicesPerGroup;

        inputs = fmatrix.create(inputDimension);
        outputs = (fmatrix) fmatrix.create(outputDimension);
        deltas = (fmatrix) fmatrix.create(outputDimension);
        errors = fmatrix.create(outputDimension);

        weights = fmatrix.create(new Dimension(inputDimension.r, inputDimension.c, inputDimension.s, 1));
        biases = fmatrix.create(new Dimension(outputDimension.r, outputDimension.c, outputDimension.s, 1));
        weights.applyFunction(x -> 1);

        deltaWeight = fmatrix.create(new Dimension(inputDimension.r, inputDimension.c, inputDimension.s, 1));
        deltaBias = fmatrix.create(new Dimension(outputDimension.r, outputDimension.c, outputDimension.s, 1));

        weightBatch = fmatrix.create(inputDimension);
        biasesBatch = fmatrix.create(outputDimension);

        weightVector = new fmatrix(weights.getHyperSliceSize(), 1);
        weightVector.applyFunction(x -> -1);

        biasVector = new fmatrix(biases.getHyperSliceSize(), 1);
        biasVector.applyFunction(x -> -1);

        weightGradient = new AdamGradientAlgorithm(weights);
        biasGradient = new AdamGradientAlgorithm(biases);

        this.function = function;
    }

    /**
     * Duplicates this layer.
     *
     * @return the duplicated layer.
     */
    @Override
    public ILayer duplicate() {
        return new PancakeLayer(inputDimension, useBias, this.slicesPerGroup, function);
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
        return inputs.getHyperSliceSize();
    }

    @Override
    public int getNrOfOutputs() {
        return outputs.getHyperSliceSize();
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return function;
    }

    @Override
    public void forward() {
        fmatrix.forwardPancake(inputs, slicesPerGroup, this.weights, this.biases, this.outputs);
        fmatrix.applyActivation(function, outputs);
        //outputs.sync();
    }

    @Override
    public void setInputs(imatrix input) {
        fmatrix.copyIntoSlice(input, this.inputs);
    }

    @Override
    public imatrix getInputs() {
        return inputs;
    }

    @Override
    public void setIdeal(imatrix ideals) {

    }

    @Override
    public imatrix getErrors() {
        return errors;
    }

    @Override
    public imatrix getOutputs() {
        return outputs;
    }

    @Override
    public void backpropagate(float learningRate) {
        //errors.sync();
        fmatrix.copyInto(outputs, deltas);
        fmatrix.applyDerivedActivation(function, deltas);
        fmatrix.dotmultiply(deltas, deltas, errors);

        calculateNewWeights(learningRate);

    }

    @Override
    public void calculateNewWeights(float learningRate) {
        //fmatrix.deltasPancake(inputs, deltas, slicesPerGroup, weightBatch, biasesBatch);
        //fmatrix.batchLC(weightBatch, this.weightVector, deltaWeight);
        //fmatrix.batchLC(biasesBatch, this.biasVector, deltaBias);
    }

    @Override
    public void calculateErrors(imatrix errors) {
        fmatrix.backpropPancake(deltas, weights, slicesPerGroup, errors);
        //errors.sync();
    }

    @Override
    public void adaptWeights(float factor) {
        //this.weightGradient.adaptWeights(deltaWeight, factor);
        //this.biasGradient.adaptWeights(deltaBias, factor);
    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {
        //fmatrix.randomize(weights, r, min, max);
        //fmatrix.randomize(biases, r, min, max);
    }

    @Override
    public void analyzeWeights() {
        WeightAnalysis wa1 = WeightAnalyzer.analyzeMatrix(this.weights);
        System.out.println("weight analysis of " + getName() + " weights");
        System.out.println(wa1);

        WeightAnalysis wa2 = WeightAnalyzer.analyzeMatrix(this.biases);
        System.out.println("weight analysis of " + getName() + " biases");
        System.out.println(wa2);
    }

    @Override
    public void writeWeightImage(String file) {
        Path wp = Paths.get(file + "_weights");
        fmatrix.writeAs3DImage(weights, 2, 5, wp);

        Path bp = Paths.get(file + "_biases");
        fmatrix.writeAs3DImage(biases, 2, 5, bp);
    }

    @Override
    public void writeOutputImage(String file) {

    }

    @Override
    public void sync() {
        this.weights.sync();
        this.biases.sync();
    }
}
