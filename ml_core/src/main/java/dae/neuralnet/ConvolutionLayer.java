package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.zpmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.cost.CostFunction;
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

    private String name;
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
     * The batch collection of weights
     */
    private final fmatrix batchWeights;
    /**
     * The inputs for this layer.
     */
    private final imatrix inputs;

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
     * Creates a new convolution layer. The inputs of the convolutional layer
     * are interpreted as a 2D array of inputs.
     *
     * @param wInputs The number of inputs in the x direction.
     * @param hInputs The number of inputs in the y direction.
     * @param features The number of convolutional layers.
     * @param stride the stride to slide the filter with.
     * @param filter the size of filter. The total number of weights per
     * convolutional layer will be filter x filter.
     * @param af the activation function.
     */
    public ConvolutionLayer(int wInputs, int hInputs, int features, int filter, int stride, ActivationFunction af) {
        this(wInputs, hInputs, 1, features, filter, stride, af);
    }

    /**
     * Creates a new convolution layer. The inputs of the convolutional layer
     * are interpreted as a 3D array of two dimensional slices.
     *
     * @param wInputs The number of inputs in the x direction.
     * @param hInputs The number of inputs in the y direction.
     * @param sInputs The number of slices in the input matrix.
     * @param features The number of convolutional layers.
     * @param stride the stride to slide the filter with.
     * @param filter the size of filter. The total number of weights per
     * convolutional layer will be filter x filter.
     * @param af the activation function.
     */
    public ConvolutionLayer(int wInputs, int hInputs, int sInputs, int features, int filter, int stride, ActivationFunction af) {
        // filter weights are shared.

        weights = new fmatrix(filter, filter, features);
        newWeights = new fmatrix(filter, filter, features);
        batchWeights = new fmatrix(filter, filter, features);

        int padding = (filter - 1) / 2;
        this.filterSize = filter;
        this.stride = stride;
        this.features = features;

        inputs = new fmatrix(wInputs, hInputs, sInputs, padding);
        backpropErrors = new fmatrix(wInputs, hInputs, sInputs);

        errors = new fmatrix(wInputs, hInputs, features);

        int oR = 1 + (wInputs - filter + padding * 2) / stride;
        int oC = 1 + (hInputs - filter + padding * 2) / stride;

        outputs = new fmatrix(oR, oC, features);
        outputVector = new fmatrix(1, outputs.getSize());
        deltas = new fmatrix(oR, oC, features);
        zpDeltas = new zpmatrix(deltas, padding);

        derivatives = new fmatrix(oR, oC, features);
        function = af;
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
                fmatrix.softMaxPerRow(outputs);
                break;
            default:
                outputs.applyFunction(this.function.getActivation());
        }
        fmatrix.matrixToRowVector(outputs, outputVector);
    }

    @Override
    public void setInputs(imatrix input) {
        if (fmatrix.equalDimension(input, this.inputs)) {
            fmatrix.copyInto(input, this.inputs);
        } else {
            if (input.isRowVector()) {
                fmatrix.rowVectorToMatrix(input, this.inputs);
            }
        }
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        fmatrix.batchConvolve(inputs, deltas, this.stride, newWeights);
        fmatrix.dotadd(batchWeights, batchWeights, newWeights);
    }

    @Override
    public void calculateErrors(fmatrix errors) {
        // errors is a row vector.
        // do correlation into properly dimensioned and zeropadded
        // backpropErrors matrix.
        fmatrix.batchBackpropCorrelate(this.zpDeltas, this.weights, this.stride, backpropErrors);
        fmatrix.matrixToRowVector(backpropErrors, errors);
    }

    @Override
    public void adaptWeights(float factor) {
        fmatrix.dotadd(weights, 1, weights, -factor, batchWeights);
        batchWeights.reset();
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
                    bi.setRGB(c, r + slice * (weights.getNrOfRows() + 5), (pi << 16) + (pi << 8) + pi);
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
    public void writeOutputImage(String file) {
        float max = this.outputs.max().value;
        float min = this.outputs.min().value;

        float factor = 255f / (max - min);
        System.out.println("factor: " + factor);

        BufferedImage bi = new BufferedImage(outputs.getNrOfColumns(), (outputs.getNrOfRows() + 5) * outputs.getNrOfSlices(), BufferedImage.TYPE_BYTE_GRAY);
        for (int slice = 0; slice < outputs.getNrOfSlices(); ++slice) {
            for (int r = 0; r < outputs.getNrOfRows(); ++r) {
                for (int c = 0; c < outputs.getNrOfColumns(); ++c) {
                    float p = outputs.get(r, c, slice);
                    int pi = (int) Math.round((p - min) * factor);
                    bi.setRGB(c, r + slice * (outputs.getNrOfRows() + 5), (pi << 16) + (pi << 8) + pi);
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
