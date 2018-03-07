package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.tmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.cost.CostFunction;
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
    private final imatrix batchDeltaWeights;
    
    private float dropRate = 0;
    private boolean dropRateSet = false;
    private Random dropRandom;
    private imatrix dropWeightMatrix;
    private imatrix tDropWeightMatrix;

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
        this.batchDeltaWeights = weightMatrixFactory.create(nrOfInputs, nrOfBiases, nrOfOutputs);
    }

    /**
     * Sets the dropRate for the weights in this layer.
     *
     * @param dropRate the drop rate for the weights
     */
    public void setDropRate(float dropRate) {
        this.dropRate = dropRate;
        this.dropRateSet = true;
        this.dropRandom = new Random(System.currentTimeMillis());
        this.constraint = new fmatrix(getNrOfInputs() + getNrOfBiases(), getNrOfOutputs());
        this.dropWeightMatrix = new fmatrix(getNrOfInputs() + getNrOfBiases(), getNrOfOutputs());
        this.tDropWeightMatrix = new tmatrix(dropWeightMatrix);
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
        imatrix weightMatrix = weights;
        if (dropRateSet) {
            this.constraint.applyFunction(x -> dropRandom.nextFloat() > dropRate ? 1 : 0);
            fmatrix.dotmultiply(dropWeightMatrix, constraint, weights);
            weightMatrix = dropWeightMatrix;
        }
        
        fmatrix.sgemm(1, inputs, weightMatrix, 0, outputs);
        
        switch (function) {
            case SOFTMAX:
                fmatrix.softMaxPerRow(outputs);
                break;
            
            default:
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
        fmatrix.sgemm(-1, tinputs, deltas, 0, deltaWeights);
        if (dropRateSet) {
            fmatrix.dotmultiply(deltaWeights, deltaWeights, constraint);
        }
        fmatrix.dotadd(batchDeltaWeights, batchDeltaWeights, deltaWeights);
    }

    /**
     * Calculates the errors that can be used in a previous layer.
     *
     * @param errors the error matrix with dimension (batchSize x nrOfInputs).
     */
    @Override
    public void calculateErrors(fmatrix errors) {
        errors.reset();
        imatrix t = tweights;
        if (dropRateSet) {
            t = tDropWeightMatrix;
        }
        fmatrix.sgemm(1, this.deltas, t, 0, errors);
        //fmatrix.multiply(deltas, this.deltas, this.tweights);
    }
    
    @Override
    public void adaptWeights(float factor) {
        fmatrix.dotadd(weights, 1, weights, factor, batchDeltaWeights);
        batchDeltaWeights.applyFunction(x -> 0.0f);
    }

    /**
     * Randomize all the weights.
     *
     * @param r the Random object to use.
     * @param min the minimum value for the random weight.
     * @param max the maximum value for the random weight.
     */
    @Override
    public void randomizeWeights(Random r, float min, float max) {
        weights.applyFunction(x -> min + r.nextFloat() * (max - min));
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
                bi.setRGB(c, r, (pi << 16) + (pi << 8) + pi);
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
        
    }
}
