package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.fsubmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
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
 * @author DAE_DEMO_BEAST
 */
public class ConvolutionLayer implements ILayer {

    /**
     * The number of different features we want to detect, this corresponds into
     * the number of layers.
     */
    private final int features;
    /**
     * The size of the rectangular filter.
     */
    private final int filterSize;
    /**
     * The stride of the convolution layer.
     */
    private final int stride;

    /**
     * The weight matrices
     */
    private final fmatrix weights;
    /**
     * the new weights.
     */
    private final fmatrix newWeights;
    /**
     * The inputs for this layer.
     */
    private final imatrix inputs;
    /**
     * The padded inputs.
     */
    private final fmatrix paddedInputs;
    /**
     * The outputs of this layer.
     */
    private final fmatrix outputs;
    private final fmatrix outputVector;
    /**
     * The errors of this layer.
     */
    private final fmatrix errors;
    /**
     * The deltas for this layer.
     */
    private final fmatrix deltas;
    /**
     * The derivatives of the output.
     */
    private final fmatrix derivatives;

    private final ActivationFunction function;

    /**
     * Creates a new convolution layer. The inputs of the convolutional layer
     * are interpreted as a 2D array of inputs.
     *
     * @param wInputs The number of inputs in the x direction.
     * @param hInputs The number of inputs in the y direction.
     * @param features The number of convolutional layers.
     * @param stride the stride to slide the filter with.
     * @param filter the size of filter. The total number of weights per
     * convolutional layer will be filter x filter.
     * @param batchSize the batchSize the use.
     * @param af the activation function.
     */
    public ConvolutionLayer(int features, int wInputs, int hInputs, int filter, int stride, int batchSize, ActivationFunction af) {
        // filter weights are shared.
        weights = new fmatrix(filter, filter, features);
        newWeights = new fmatrix(filter, filter, features);

        int padding = (filter - 1) / 2;
        this.filterSize = filter;
        this.stride = stride;
        this.features = features;

        paddedInputs = new fmatrix(wInputs, hInputs, 1, padding);
        inputs = new fsubmatrix(paddedInputs, padding, padding, hInputs, wInputs);

        errors = new fmatrix(wInputs, hInputs, features);

        int oR = 1 + (wInputs - filter + padding * 2) / stride;
        int oC = 1 + (hInputs - filter + padding * 2) / stride;

        outputs = new fmatrix(oR, oC, features);
        outputVector = new fmatrix(1, outputs.getSize());
        deltas = new fmatrix(oR, oC, features);

        derivatives = new fmatrix(oR, oC, features);
        function = af;
    }

    @Override
    public int getNrOfInputs() {
        return inputs.getSize();
    }

    @Override
    public int getNrOfOutputs() {
        return outputVector.getSize();
    }

    public int getNrOfFeatures() {
        return features;
    }

    @Override
    public void forward() {
        fmatrix.batchConvolve(inputs, this.weights, stride, this.outputs);
        switch (function) {
            case SOFTMAX:
                outputs.softMaxPerRow();
                break;
            default:
                outputs.applyFunction(this.function.getActivation());
        }
        fmatrix.matrixToRowVector(outputs, outputVector);
    }

    @Override
    public void setInputs(imatrix input) {
        fmatrix.rowVectorToMatrix(input, this.inputs);

    }

    @Override
    public void calculateNewWeights(float learningRate) {
        fmatrix.copyInto(weights, newWeights);
        fmatrix.batchCorrelate(inputs, deltas, 1, newWeights);
        newWeights.multiply(learningRate);
    }

    @Override
    public void calculateErrors(fmatrix errors) {
        // errors is a row vector.
        // first do the flipped convolution.

    }

    @Override
    public void adaptWeights() {
        fmatrix.dotsubtract(weights, newWeights);
    }

    @Override
    public void writeWeightImage(String file) {
        float max = weights.max().value;
        float min = weights.min().value;

        float factor = 255f / (max - min);
        System.out.println("factor: " + factor);

        imatrix weightCopy = weights.copy();
        weightCopy.applyFunction(x -> (x - min) * factor);

        BufferedImage bi = new BufferedImage(weights.getNrOfColumns(), (weights.getNrOfRows() + 5) * this.getNrOfFeatures(), BufferedImage.TYPE_BYTE_GRAY);
        for (int slice = 0; slice < this.getNrOfFeatures(); ++slice) {
            for (int r = 0; r < weightCopy.getNrOfRows(); ++r) {
                for (int c = 0; c < weightCopy.getNrOfColumns(); ++c) {
                    float p = weightCopy.get(r, c, slice);
                    int pi = (int) Math.round(p);
                    bi.setRGB(c, r + slice * (weights.getNrOfRows()+5), (pi << 16) + (pi << 8) + pi);
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
    public void backpropagate(float learningRate, boolean calculateErrors) {
        // 2. multiply with derivative of activation function. 
        // Note: derivative is f'(net input) but this is typically expressed
        // in terms of the output of the activation function.
        // 2.a copy the outputs into the derivatives matrix.
        fmatrix.copyInto(this.outputs, this.derivatives);
        // 2.b apply the derivative of the activation function to the output.
        derivatives.applyFunction(function.getDerivedActivation());

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
        System.out.println("Start weights: ");
        System.out.println(weights);
    }

    @Override
    public void setIdeal(imatrix ideals) {

    }

    @Override
    public fmatrix getErrors() {
        return this.errors;
    }

    @Override
    public fmatrix getOutputs() {
        return outputVector;
    }

}
