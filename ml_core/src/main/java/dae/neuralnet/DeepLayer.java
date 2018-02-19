package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import java.util.Calendar;
import java.util.Date;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class DeepLayer {

    private final AbstractLayer[] layers;
    private boolean validNetwork;

    /**
     * Creates a new deep layer neural network with the given inputs and outputs
     * and
     *
     * @param layers the layers that will form the neural network.
     */
    public DeepLayer(AbstractLayer... layers) {
        this.layers = layers;
        validNetwork = true;
        for (int i = 0; i < (layers.length - 1); ++i) {
            AbstractLayer current = layers[i];
            AbstractLayer next = layers[i + 1];
            if (current.getNrOfOutputs() != next.getNrOfInputs()) {
                validNetwork = false;
                throw new IllegalArgumentException("Error while constructing a DeepLayer object.\n"
                        + "Outputs of layer " + i + " do not match inputs of the next layer.");
            }
        }

    }

    public void randomizeWeights() {
        for (AbstractLayer l : layers) {
            l.randomizeWeights();
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
    public AbstractLayer getFirstLayer() {
        return layers[0];
    }

    /**
     * Get the last layer in the deep learning system.
     *
     * @return the last layer.
     */
    public AbstractLayer getLastLayer() {
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
     * @param learningRate
     * @param input the training input.
     * @param target the target output.
     */
    public void train(float learningRate, imatrix input, imatrix target) {
        setTarget(target);
        forward(input);

        for (int i = layers.length; i > 0; --i) {
            AbstractLayer current = layers[i - 1];
            // only calculate deltas for last layer.
            current.backpropagate(learningRate, i == layers.length);

            if ((i - 2) >= 0) {
                AbstractLayer previous = layers[i - 2];
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
        int  month = c.get(Calendar.MONTH);
        int day = c.get(Calendar.DAY_OF_MONTH);
        int hour = c.get(Calendar.HOUR_OF_DAY);
        int mins = c.get(Calendar.MINUTE);
        for (AbstractLayer l : this.layers) {
            l.writeWeightImage("weight_"+year+"_"+month+"_"+day+"#"+hour+"_"+mins+"_"+layerIndex);
            ++layerIndex;
        }
    }
}
