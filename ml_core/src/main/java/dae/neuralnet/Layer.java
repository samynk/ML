package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.tmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.matrix.MatrixFactory;
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
 * The neural network layer is the basic interface that defines the
 * functionality of a neural network layer. A layer has a fixed number of
 * variable inputs, bias nodes and output nodes.
 *
 * The weights of the layer can be a matrix that has the dimension (inputs+bias)
 * x (outputs), but other configurations are also allowed (as for example in
 * convolutional neural networks).
 *
 * @author Koen Samyn
 */
public class Layer extends AbstractLayer {

    private final imatrix weights;
    private final imatrix tweights;
    private imatrix constraint;

    private final imatrix deltaWeights;

    /**
     * Creates a layer with a given number of inputs, biases and outputs.
     *
     * The number of inputs n is defined as: nrOfInputs+nrOfBiases.
     *
     * The input matrix is then defined as an (1 x n) matrix. The bias values
     * will be set to one.
     *
     * The number of outputs m is equal to the nrOfOutputs parameter.
     *
     * The weight matrix is thus defined as an (n x m) matrix.
     *
     * Multiplication of the (1xn) matrix with the (n x m) matrix leads to an
     * output matrix with dimension (1 x m).
     *
     * @param nrOfInputs the number of inputs of this layer.
     * @param nrOfBiases the number of biases of this layer.
     * @param nrOfOutputs the number of outputs of this layer.
     * @param af the activation function
     */
    public Layer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, ActivationFunction af) {
        this(nrOfInputs, nrOfBiases, nrOfOutputs, 1, af, MatrixFactory.DEFAULT);
    }

    /**
     * Creates a layer with a given number of inputs, biases and outputs.
     *
     * The number of inputs n is defined as: nrOfInputs+nrOfBiases.
     *
     * The input matrix is then defined as an (batchSize x n) matrix. The bias
     * values will be set to one.
     *
     * The number of outputs m is equal to the nrOfOutputs parameter.
     *
     * The weight matrix is thus defined as an (n x m) matrix.
     *
     * Multiplication of the (batchSizexn) matrix with the (n x m) matrix leads
     * to an output matrix with dimension (batchSize x m).
     *
     * @param nrOfInputs the number of inputs of this layer.
     * @param nrOfBiases the number of biases of this layer.
     * @param nrOfOutputs the number of outputs of this layer.
     * @param batchSize the batch size
     * @param af the activation function
     */
    public Layer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, int batchSize, ActivationFunction af) {
        this(nrOfInputs, nrOfBiases, nrOfOutputs, batchSize, af, MatrixFactory.DEFAULT);
    }

    /**
     * Creates a layer with a given number of inputs, biases and outputs.
     *
     * The number of inputs n is defined as: nrOfInputs+nrOfBiases.
     *
     * The input matrix is then defined as an (1 x n) matrix. The bias values
     * will be set to one.
     *
     * The number of outputs m is equal to the nrOfOutputs parameter.
     *
     * The weight matrix is thus defined as an (n x m) matrix.
     *
     * Multiplication of the (1xn) matrix with the (n x m) matrix leads to an
     * output matrix with dimension (1 x m).
     *
     * @param nrOfInputs the number of inputs of this layer.
     * @param nrOfBiases the number of biases of this layer.
     * @param nrOfOutputs the number of outputs of this layer.
     * @param batchSize the batch size for this layer.
     * @param af the activation function
     * @param weightMatrixFactory the factory class that creates matrices.
     */
    public Layer(int nrOfInputs, int nrOfBiases, int nrOfOutputs, int batchSize, ActivationFunction af, MatrixFactory weightMatrixFactory) {
        super(nrOfInputs, nrOfBiases, nrOfOutputs, batchSize, af);
        this.weights = weightMatrixFactory.create(nrOfInputs, nrOfBiases, nrOfOutputs);
        this.tweights = new tmatrix(weights);
        this.deltaWeights = weightMatrixFactory.create(nrOfInputs, nrOfBiases, nrOfOutputs);
    }

    /**
     * A matrix that constraints the weights that can be changed.
     *
     * @param constraint the constraint matrix.
     */
    public void setConstraint(imatrix constraint) {
        this.constraint = constraint;
    }

    /**
     * Returns the matrix with the weights of the single neural network layer.
     *
     * @return the weights of the neural net.
     */
    public imatrix getWeights() {
        return this.weights;
    }

    /**
     * Returns the deltas for the weights in this layer.
     *
     * @return the matrix with the delta weights.
     */
    public imatrix getDeltaWeights() {
        return this.deltaWeights;
    }

    /**
     * Multiplies the input matrix with the weight matrix and stores the result
     * in the output matrix.
     */
    @Override
    public void forward() {
        outputs.reset();
        fmatrix.sgemm(1, inputs, weights, 0, outputs);

        // fmatrix.multiply(outputs, inputs, weights);
        //outputs.applyFunction(this.activation);
        if (this.function == ActivationFunction.SOFTMAX) {
            // row wise softmax.
            outputs.softMaxPerRow();
        } else {
            outputs.applyFunction(this.activation);
        }
    }

    @Override
    public void calculateNewWeights(float learningRate) {
//        // 4.a Multiply the transposes inputs with the deltas.
//        fmatrix.multiply(this.deltaWeights, this.tinputs, this.deltas);
//
//        // 4.b Multiply with the learning rate.
//        deltaWeights.multiply(learningRate);
//        // 4.c apply the constraints
//        if (constraint != null) {
//            fmatrix.dotmultiply(deltaWeights, deltaWeights, constraint);
//        }
//        // 4.d deltaWeights now holds the new values for the weights.
//        fmatrix.dotsubtract(deltaWeights, weights, deltaWeights);
        fmatrix.copyInto(weights, deltaWeights);
        fmatrix.sgemm(-learningRate / getBatchSize(), tinputs, deltas, 1, deltaWeights);
    }

    /**
     * Calculates the errors that can be used in a previous layer.
     *
     * @param errors the error matrix with dimension (batchSize x nrOfInputs).
     */
    @Override
    public void calculateErrors(fmatrix errors) {
        errors.reset();
        fmatrix.sgemm(1, this.deltas, tweights, 0, errors);
        //fmatrix.multiply(deltas, this.deltas, this.tweights);
    }

    @Override
    public void adaptWeights() {
        fmatrix.copyInto(deltaWeights, weights);
    }

    @Override
    public void randomizeWeights() {
        Random r = new Random();
        weights.applyFunction(x -> (r.nextFloat()-.5f) * 2);
    }

    public void printInputs() {
        System.out.println(inputs.toString());
    }

    @Override
    public void writeWeightImage(String file) {
        float max = weights.max().value;
        float min = weights.min().value;
        
        float factor = 255f / (max - min);
        System.out.println("factor: " + factor);
        
        imatrix weightCopy = weights.copy();
        weightCopy.applyFunction(x -> (x - min) * factor);

        BufferedImage bi = new BufferedImage(weights.getNrOfColumns(), weights.getNrOfRows(), BufferedImage.TYPE_BYTE_GRAY);
        for (int r = 0; r < weightCopy.getNrOfRows(); ++r) {
            for (int c = 0; c < weightCopy.getNrOfColumns(); ++c) {
                float p = weightCopy.get(r, c);
                int pi = (int) Math.round(p);
                bi.setRGB(c, r, (pi << 16) + (pi << 8) +  pi);
            }
        }
        
        String homeDir = System.getProperty("user.home");
        Path exportPath = Paths.get(homeDir,".nn",file+".png");
        try {
            Files.createDirectories(exportPath);
            ImageIO.write(bi, "png",exportPath.toFile());
        } catch (IOException ex) {
            Logger.getLogger(Layer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
