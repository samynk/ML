/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
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
 * This layer pools the inputs with the given filter. A pooling layer can be
 * configured as part of a convolutional layer.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class PoolLayer implements ILayer {

    private String name;
    private int scaleX, scaleY;

    private final fmatrix inputs;
    /**
     * The mask layer helps to propagate the deltas to the previous layer.
     */
    private final intmatrix maskLayer;
    /**
     * The output matrix.
     */
    private final fmatrix outputs;

    private final fmatrix deltas;
    /**
     * The errors to back propagate in a previous layer.
     */
    private final fmatrix errors;

    public PoolLayer(int iWidth, int iHeight, int iSlices, int scaleX, int scaleY) {
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        inputs = new fmatrix(iHeight, iWidth, iSlices);
        maskLayer = new intmatrix(iHeight / scaleY, iWidth / scaleX, iSlices);
        outputs = new fmatrix(iHeight / scaleY, iWidth / scaleX, iSlices);
        deltas = new fmatrix(iHeight / scaleY, iWidth / scaleX, iSlices);
        errors = new fmatrix(1, deltas.getSize());
    }

    /**
     * Returns the activation function for this layer.
     *
     * @return the activation function.
     */
    @Override
    public ActivationFunction getActivationFunction() {
        return ActivationFunction.IDENTITY;
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
        return outputs.getSize();
    }

    public int getNrofWInputs() {
        return inputs.getNrOfColumns();
    }

    public int getNrOfHInputs() {
        return inputs.getNrOfRows();
    }

    public int getNrOfSInputs() {
        return inputs.getNrOfSlices();
    }

    public int getScaleX() {
        return scaleX;
    }

    public int getScaleY() {
        return scaleY;
    }

    @Override
    public void forward() {
        fmatrix.batchMaxPool(inputs, outputs, maskLayer);
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
        // no weights to adapt
        fmatrix.rowVectorToMatrix(errors, deltas);
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        // no weights in this layer.
    }

    /**
     * The error matrix is the errors of the previous layer that needs to be
     * calculated.
     *
     * @param errors
     */
    @Override
    public void calculateErrors(fmatrix errors) {
        fmatrix.batchBackpropMaxPool(deltas, this.maskLayer, errors);
    }

    @Override
    public void adaptWeights(float factor) {
        // no weights in this layer.
    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {
        // no weights in this layer.
    }

    @Override
    public void writeWeightImage(String file) {
        // not supported
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
    public void analyzeWeights() {
        System.out.println("No weights to analyze " + this.getName());
    }

}
