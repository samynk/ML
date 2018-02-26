package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import java.util.Calendar;
import java.util.Random;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class DeepLayer {

    private final ILayer[] layers;
    private boolean validNetwork;
    private final LearningRate learningRate;

    /**
     * Creates a new deep layer neural network with the given inputs and outputs
     * and
     *
     * @param lr the learning rate calculator for this neural network.
     * @param layers the layers that will form the neural network.
     */
    public DeepLayer(LearningRate lr, ILayer... layers) {
        this.learningRate = lr;
        this.layers = layers;
        validNetwork = true;
        for (int i = 0; i < (layers.length - 1); ++i) {
            ILayer current = layers[i];
            ILayer next = layers[i + 1];
            if (current.getNrOfOutputs() != next.getNrOfInputs()) {
                validNetwork = false;
                throw new IllegalArgumentException("Error while constructing a DeepLayer object.\n"
                        + "Outputs of layer " + i + " do not match inputs of the next layer.");
            }
        }

    }

    /**
     * Randomize all the weights.
     *
     * @param r the Random object to use.
     * @param min the minimum value for the random weight.
     * @param max the maximum value for the random weight.
     */
    public void randomizeWeights(Random r, float min, float max) {
        for (ILayer l : layers) {
            l.randomizeWeights(r, min, max);
        }
    }

    /**
     * Checks if the network structure is valid.
     *
     * @return true if the network structure is valid, false otherwise.
     */
    public boolean isValid() {
        return validNetwork;
    }

    /**
     * Get the first layer in the deep learning system.
     *
     * @return the first layer.
     */
    public ILayer getFirstLayer() {
        return layers[0];
    }

    /**
     * Get the last layer in the deep learning system.
     *
     * @return the last layer.
     */
    public ILayer getLastLayer() {
        return layers[layers.length - 1];
    }

    /**
     * Sets the inputs for this deep learning system.
     *
     * @param input the inputs of the system.
     */
    public void setInputs(imatrix input) {
        getFirstLayer().setInputs(input);
    }

    public void setTarget(imatrix target) {
        getLastLayer().setIdeal(target);
    }

    /**
     *
     * @param iteration the current iteration in the training phase.
     * @param input the training input.
     * @param target the target output.
     */
    public void train(int iteration, imatrix input, imatrix target) {
        setTarget(target);
        forward(input);
        float lr = this.learningRate.getLearningRate(iteration);

        for (int i = layers.length; i > 0; --i) {
            ILayer current = layers[i - 1];
            // only calculate deltas for last layer.
            current.backpropagate(lr, i == layers.length);

            if ((i - 2) >= 0) {
                ILayer previous = layers[i - 2];
                current.calculateErrors(previous.getErrors());
            }
        }

        for (int i = 0; i < layers.length; ++i) {
            layers[i].adaptWeights();
        }
    }

    public void forward(imatrix input) {
        setInputs(input);
        for (int i = 0; i < layers.length; ++i) {
            layers[i].forward();

            if (i + 1 < layers.length) {
                fmatrix outputs = layers[i].getOutputs();
                layers[i + 1].setInputs(outputs);
            }
        }
    }

    /**
     * Writes a weight representation of every weight.
     */
    public void writeWeightImages() {
        int layerIndex = 1;
        Calendar c = Calendar.getInstance();
        int year = c.get(Calendar.YEAR);
        int month = c.get(Calendar.MONTH);
        int day = c.get(Calendar.DAY_OF_MONTH);
        int hour = c.get(Calendar.HOUR_OF_DAY);
        int mins = c.get(Calendar.MINUTE);
        for (ILayer l : this.layers) {
            l.writeWeightImage("weight_" + year + "_" + month + "_" + day + "#" + hour + "_" + mins + "_" + layerIndex);
            ++layerIndex;
        }
    }
}
