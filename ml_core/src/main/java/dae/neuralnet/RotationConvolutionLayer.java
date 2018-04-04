/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.fmatrixview;
import dae.matrix.imatrix;
import dae.matrix.zpmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.analysis.WeightAnalysis;
import dae.neuralnet.analysis.WeightAnalyzer;
import dae.neuralnet.gradient.AdamGradientAlgorithm;
import dae.neuralnet.gradient.GradientAlgorithm;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class RotationConvolutionLayer implements ILayer {

    private String name;
    /**
     * The number of different features we want to detect, this corresponds into
     * the number of layers.
     */
    private final int features;
    /**
     * The number of rotations per feature.
     */
    private final int rotations;
    /**
     * The size of the rectangular filter.
     */
    private final int filterSize;
    /**
     * The stride of the convolution layer.
     */
    private final int stride;
    /**
     * The batch size of the convolutional layer.
     */
    private final int batchSize;
    /**
     * The weight matrices
     */
    private final imatrix weights;
    /**
     * the new weights.
     */
    private final fmatrix newWeights;
    /**
     * The inputs for this layer.
     */
    private final imatrix inputs;

    /**
     * The outputs of this layer, before selecting the maximum
     * rotational activation.
     */
    private final fmatrix intermediateOutputs;
    /**
     * The outputs of this layer, the maximum activation is stored in
     * the odd numbered slices, the rotational value in the even numbered
     * slices.
     */
    private final fmatrix outputs;
    private final fmatrix outputVector;
    /**
     * The errors of this layer.
     */
    private final fmatrix errors;
    private final imatrix flatErrorView;
    /**
     * The deltas for this layer.
     */
    private final fmatrix deltas;
    /**
     * The deltas with zero padding.
     */
    private final imatrix zpDeltas;
    /**
     * The derivatives of the output.
     */
    private final fmatrix derivatives;

    /**
     * A utility matrix to store the backpropagation result.
     */
    private final fmatrix backpropErrors;

    private final ActivationFunction function;

    /**
     * The gradient algorithm.
     */
    private GradientAlgorithm gradientAlgorithm;
    
    private float startAngle = 0;
    private float endAngle = (float) Math.PI;

    /**
     * Creates a new convolution layer. The inputs of the convolutional layer
     * are interpreted as a 2D array of inputs.
     *
     * @param wInputs The number of inputs in the x direction.
     * @param hInputs The number of inputs in the y direction.
     * @param features The number of convolutional layers.
     * @param rotations The number of rotations per feature.
     * @param stride the stride to slide the filter with.
     * @param filter the size of filter. The total number of weights per
     * convolutional layer will be filter x filter.
     * @param batchSize the batch size of the convolutional layer.
     * @param af the activation function.
     */
    public RotationConvolutionLayer(int wInputs, int hInputs, int features, int rotations, int filter, int stride, int batchSize, ActivationFunction af) {
        this(wInputs, hInputs, 1, features, rotations, filter, stride, batchSize, af);
    }

    /**
     * Creates a new convolution layer. The inputs of the convolutional layer
     * are interpreted as a 3D array of two dimensional slices.
     *
     * @param wInputs The number of inputs in the x direction.
     * @param hInputs The number of inputs in the y direction.
     * @param sInputs The number of slices in the input matrix.
     * @param features The number of convolutional layers.
     * @param rotations The number of rotations per feature.
     * @param stride the stride to slide the filter with.
     * @param filter the size of filter. The total number of weights per
     * convolutional layer will be filter x filter.
     * @param batchSize the batch size of the convolutional layer.
     * @param af the activation function.
     */
    public RotationConvolutionLayer(int wInputs, int hInputs, int sInputs, int features, int rotations, int filter, int stride, int batchSize, ActivationFunction af) {
        // filter weights are shared.
        this(wInputs, hInputs, sInputs, features, rotations, filter, stride, batchSize, af, new fmatrix(filter, filter, sInputs * features * rotations));
    }

    /**
     * Creates a new convolution layer. The inputs of the convolutional layer
     * are interpreted as a 3D array of two dimensional slices.
     *
     * @param wInputs The number of inputs in the x direction.
     * @param hInputs The number of inputs in the y direction.
     * @param sInputs The number of slices in the input matrix.
     * @param features The number of convolutional layers.
     * @param rotations The number of rotations per feature.
     * @param stride the stride to slide the filter with.
     * @param filter the size of filter. The total number of weights per
     * convolutional layer will be filter x filter.
     * @param batchSize the batch size of the convolutional layer.
     * @param af the activation function.
     * @param weights
     */
    public RotationConvolutionLayer(int wInputs, int hInputs, int sInputs, int features, int rotations, int filter, int stride, int batchSize, ActivationFunction af, imatrix weights) {
        // filter weights are shared.

        this.weights = weights;
        this.gradientAlgorithm = new AdamGradientAlgorithm(weights);
        newWeights = new fmatrix(filter, filter, sInputs * features);

        int padding = (filter - 1) / 2;
        this.filterSize = filter;
        this.stride = stride;
        this.features = features;
        this.rotations = rotations;

        inputs = new fmatrix(wInputs, hInputs, sInputs, batchSize, padding);
        backpropErrors = new fmatrix(wInputs, hInputs, sInputs, batchSize);

        errors = new fmatrix(wInputs, hInputs, sInputs * features, batchSize);
        flatErrorView = new fmatrixview(errors.getHyperSliceSize(), 1, 1, errors);

        int oR = 1 + (wInputs - filter + padding * 2) / stride;
        int oC = 1 + (hInputs - filter + padding * 2) / stride;

        intermediateOutputs = new fmatrix(oR, oC, sInputs * features * rotations, batchSize);
        outputs = new fmatrix(oR, oC, sInputs * features * 2, batchSize);
        outputVector = new fmatrix(intermediateOutputs.getHyperSliceSize(), 1, 1, batchSize);
        deltas = new fmatrix(oR, oC, sInputs * features, batchSize);
        zpDeltas = new zpmatrix(deltas, padding);

        derivatives = new fmatrix(oR, oC, sInputs * features, batchSize);
        function = af;

        this.batchSize = batchSize;
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return this.function;
    }

    /**
     * Set the name of this layer.
     *
     * @param name the name of the layer.
     */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the name of this layer.
     *
     * @return the name of the layer.
     */
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
        return outputVector.getHyperSliceSize();
    }

    public int getNrOfFeatures() {
        return features;
    }

    public int getNrOfWInputs() {
        return inputs.getNrOfRows();
    }

    public int getNrOfHInputs() {
        return inputs.getNrOfColumns();
    }

    public int getNrOfSInputs() {
        return inputs.getNrOfSlices();
    }

    public int getFilterSize() {
        return this.filterSize;
    }

    public int getFilterStride() {
        return this.stride;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public imatrix getWeights() {
        return this.weights;
    }

    @Override
    public void forward() {
        fmatrix.rotateKernels(this.weights, features, rotations, startAngle, endAngle);
        fmatrix.batchConvolve(inputs, this.weights, stride, this.intermediateOutputs);

        switch (function) {
            case SOFTMAX:
                fmatrix.softMaxPerRow(intermediateOutputs);
                break;
            default:
                intermediateOutputs.applyFunction(this.function.getActivation());
        }
        fmatrix.copyIntoSlice(intermediateOutputs, outputVector);
    }

    @Override
    public void setInputs(imatrix input) {
        fmatrix.copyIntoSlice(input, this.inputs);
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        fmatrix.batchConvolve(inputs, deltas, this.stride, newWeights);
    }

    @Override
    public void calculateErrors(imatrix errors) {
        // errors is a row vector.
        // do correlation into properly dimensioned and zeropadded
        // backpropErrors matrix.
        fmatrix.batchBackpropCorrelate(this.zpDeltas, this.weights, this.stride, backpropErrors);
        fmatrix.copyIntoSlice(backpropErrors, errors);
    }

    @Override
    public void adaptWeights(float factor) {
        gradientAlgorithm.adaptWeights(newWeights, factor);
    }

    @Override
    public void writeWeightImage(String file) {
        Path p = Paths.get(file);
        fmatrix.writeAs3DImage(weights, (int) (Math.sqrt(features * inputs.getNrOfSlices()) + 1), 5, p);
    }

    @Override
    public void writeOutputImage(String file) {
        float max = this.intermediateOutputs.max().value;
        float min = this.intermediateOutputs.min().value;

        float factor = 255f / (max - min);
        System.out.println("factor: " + factor);

        BufferedImage bi = new BufferedImage(intermediateOutputs.getNrOfColumns(), (intermediateOutputs.getNrOfRows() + 5) * intermediateOutputs.getNrOfSlices(), BufferedImage.TYPE_BYTE_GRAY);
        for (int slice = 0; slice < intermediateOutputs.getNrOfSlices(); ++slice) {
            for (int r = 0; r < intermediateOutputs.getNrOfRows(); ++r) {
                for (int c = 0; c < intermediateOutputs.getNrOfColumns(); ++c) {
                    float p = intermediateOutputs.get(r, c, slice);
                    int pi = (int) Math.round((p - min) * factor);
                    bi.setRGB(c, r + slice * (intermediateOutputs.getNrOfRows() + 5), (pi << 16) + (pi << 8) + pi);
                }
            }
        }

        String homeDir = System.getProperty("user.home");
        Path exportPath = Paths.get(homeDir, ".nn", file + ".png");
        try {
            Files.createDirectories(exportPath);
            ImageIO.write(bi, "png", exportPath.toFile());
        } catch (IOException ex) {
            Logger.getLogger(Layer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void backpropagate(float learningRate) {
        // 2. multiply with derivative of activation function. 
        // Note: derivative is f'(net input) but this is typically expressed
        // in terms of the output of the activation function.
        // 2.a copy the outputs into the derivatives matrix.
        fmatrix.copyInto(this.intermediateOutputs, this.derivatives);

        // 2.b apply the derivative of the activation function to the output.
        fmatrix.applyDerivedActivation(function, derivatives);

        // 1. copy output - ideal (target) into deltas.
        // todo is it necessary to have a convolution layer as last layer?
        //        if (calculateErrors) {
        //            fmatrix.dotsubtract(errors, this.outputs, this.ideal);
        //        }
        // 3. multiply the derivatives with the errors.
        fmatrix.dotmultiply(deltas, errors, derivatives);

        // 4. Calculate the new weights
        calculateNewWeights(learningRate);
    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {
        weights.applyFunction(x -> min + r.nextFloat() * (max - min));
    }

    @Override
    public void setIdeal(imatrix ideals) {

    }

    @Override
    public imatrix getErrors() {
        return this.flatErrorView;
    }

    @Override
    public fmatrix getOutputs() {
        return outputVector;
    }

    @Override
    public void analyzeWeights() {
        WeightAnalysis wa = WeightAnalyzer.analyzeMatrix(this.weights);
        System.out.println("weight analysis of " + this.name);
        System.out.println(wa);
    }

    /**
     * Syncs the matrices with the matrices on the gpu.
     */
    @Override
    public void sync() {
        this.weights.sync();
    }
}
