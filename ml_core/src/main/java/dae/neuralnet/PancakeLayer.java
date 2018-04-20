/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.Dimension;
import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.gradient.AdamGradientAlgorithm;
import dae.neuralnet.gradient.GradientAlgorithm;
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
    private int biases;
    private String name;

    private final imatrix inputs;
    private final imatrix outputs;

    private final imatrix weights;
    private final GradientAlgorithm gradient;

    private final imatrix errors;

    private ActivationFunction function;

    /**
     * Creates a new pancake layer.
     *
     * @param inputDimension the dimensions of the input.
     * @param biases the biases.
     * @param slicesPerGroup the slicesPerGroup.
     * @param function the activation function of this pancake layer.
     */
    public PancakeLayer(Dimension inputDimension, int biases, int slicesPerGroup, ActivationFunction function) {
        this.inputDimension = inputDimension;
        this.outputDimension = Dimension.Dim(
                inputDimension.getRows(),
                inputDimension.getColumns(),
                inputDimension.getSlices() / slicesPerGroup,
                inputDimension.getHyperSlices());
        this.biases = biases;

        inputs = fmatrix.create(inputDimension);
        outputs = fmatrix.create(outputDimension);
        errors = fmatrix.create(outputDimension);

        int weightSlices = slicesPerGroup * (biases + inputDimension.getSlices() / slicesPerGroup);
        Dimension wd = new Dimension(inputDimension.getRows(),
                inputDimension.getColumns(),
                weightSlices,
                inputDimension.getHyperSlices());
        weights = fmatrix.create(wd);
        gradient = new AdamGradientAlgorithm(weights);

        this.function = function;
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

    }

    @Override
    public void calculateNewWeights(float learningRate) {

    }

    @Override
    public void calculateErrors(imatrix errors) {

    }

    @Override
    public void adaptWeights(float factor) {

    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {

    }

    @Override
    public void analyzeWeights() {

    }

    @Override
    public void writeWeightImage(String file) {

    }

    @Override
    public void writeOutputImage(String file) {

    }

    @Override
    public void sync() {

    }
}
