/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.Dimension;
import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
import java.util.Random;

/**
 * This layer contains two sub layer that will receive the inputs of this layer
 * based on the slices of the input. The even slices will be sent to the first
 * layer, the uneven slices will be sent to the second sublayer.
 *
 * For now, the composite layer should not change the row and column sizes of
 * the input.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class CompositeLayer implements ILayer {

    private String name;

    private final Dimension inputDim;
    private final Dimension outputDim;
    private final ILayer layer1;
    private final ILayer layer2;

    private imatrix inputs;
    private imatrix inputs1;
    private imatrix inputs2;

    private final imatrix errors;
    private final imatrix layer1Errors;
    private final imatrix layer2Errors;

    private final imatrix outputs;

    /**
     * Creates a composite layer with the given layers.
     *
     * @param inputDim the dimension of the input.
     * @param outputDim the dimension of the output.
     * @param layer1 the first layer.
     * @param layer2 the second layer.
     */
    public CompositeLayer(Dimension inputDim, Dimension outputDim, ILayer layer1, ILayer layer2) {
        this.outputDim = outputDim;
        this.inputDim = inputDim;
        this.layer1 = layer1;
        this.layer2 = layer2;
        
        inputs1 = new fmatrix(this.inputDim.getRows(), this.inputDim.getColumns(), this.inputDim.getSlices()/2, this.inputDim.getHyperSlices());
        inputs2 = new fmatrix(this.inputDim.getRows(), this.inputDim.getColumns(), this.inputDim.getSlices()/2, this.inputDim.getHyperSlices());

        errors = fmatrix.create(outputDim);
        layer1Errors = new fmatrix(this.inputDim.getRows(), this.inputDim.getColumns(), this.inputDim.getSlices()/2, this.inputDim.getHyperSlices());
        layer2Errors = new fmatrix(this.inputDim.getRows(), this.inputDim.getColumns(), this.inputDim.getSlices()/2, this.inputDim.getHyperSlices());
        outputs = fmatrix.create(this.outputDim);
    }

    /**
     * Sets the name of the layer.
     *
     * @param name the name of the layer.
     */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the name of the layer.
     *
     * @return the name of the layer.
     */
    @Override
    public String getName() {
        return name;
    }

    /**
     * Returns the number of inputs of this layer.
     *
     * @return
     */
    @Override
    public int getNrOfInputs() {
        return layer1.getNrOfInputs() + layer2.getNrOfInputs();
    }

    /**
     * Returns the number of outputs of this layer.
     *
     * @return
     */
    @Override
    public int getNrOfOutputs() {
        return layer1.getNrOfOutputs() + layer2.getNrOfOutputs();
    }

    /**
     * Returns the activation function of this layer, currently not used.
     *
     * @return the IDENTITY activation function.
     */
    @Override
    public ActivationFunction getActivationFunction() {
        return ActivationFunction.IDENTITY;
    }

    @Override
    public void forward() {
        layer1.forward();
        layer2.forward();
        fmatrix.zip(layer1.getOutputs(), layer2.getOutputs(), outputs);
    }

    @Override
    public void setInputs(imatrix input) {
        this.inputs = input;
        
        fmatrix.unzip(input, this.inputs1,this.inputs2);
        layer1.setInputs(this.inputs1);
        layer2.setInputs(this.inputs2);
    }

    /**
     * Returns the inputs of this layer.
     *
     * @return the inputs of this layer.
     */
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
        fmatrix.unzip(errors, layer1.getErrors(), layer2.getErrors());
        
        layer1.backpropagate(learningRate);
        layer2.backpropagate(learningRate);
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        layer1.calculateNewWeights(learningRate);
        layer2.calculateNewWeights(learningRate);
    }

    @Override
    public void calculateErrors(imatrix errors) {
        layer1.calculateErrors(layer1Errors);
        layer2.calculateErrors(layer2Errors);
        fmatrix.zip(layer1Errors, layer2Errors, errors);
    }

    @Override
    public void adaptWeights(float factor) {
        layer1.adaptWeights(factor);
        layer2.adaptWeights(factor);
    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {
        layer1.randomizeWeights(r, min, max);
        layer2.randomizeWeights(r, min, max);
    }

    @Override
    public void analyzeWeights() {
        layer1.analyzeWeights();
        layer2.analyzeWeights();
    }

    @Override
    public void writeWeightImage(String file) {
        layer1.writeWeightImage(file);
        layer2.writeWeightImage(file);
    }

    @Override
    public void writeOutputImage(String file) {
        layer1.writeOutputImage(file);
        layer2.writeOutputImage(file);
    }

    @Override
    public void sync() {
        layer1.sync();
        layer2.sync();
    }

}
