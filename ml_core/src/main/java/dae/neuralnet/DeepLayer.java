package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.cost.CostFunction;
import dae.neuralnet.cost.QuadraticCostFunction;
import dae.neuralnet.io.DeepLayerMetaData;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Random;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class DeepLayer {

    private ILayer[] layers;
    private boolean validNetwork;
    private LearningRate learningRate;
    private CostFunction costFunction = new QuadraticCostFunction();

    private DeepLayerMetaData metadata = new DeepLayerMetaData();

    /**
     * Creates an empty DeepLayer object.
     *
     */
    public DeepLayer() {

    }

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
        checkLayers();
    }

    private void checkLayers() throws IllegalArgumentException {
        for (int i = 0; i < (layers.length - 1); ++i) {
            ILayer current = layers[i];
            ILayer next = layers[i + 1];
            if (current.getNrOfOutputs() != next.getNrOfInputs()) {
                int outputs = current.getNrOfOutputs();
                int inputs = next.getNrOfInputs();
                validNetwork = false;
                throw new IllegalArgumentException("Error while constructing a DeepLayer object.\n"
                        + "Outputs of layer " + layers[i].getName() + " do not match inputs of the next layer.\n" +
                        "number of inputs :" + inputs +", outputs: " + outputs+ "\n" );
            }
        }
    }

    /**
     * Returns the layers in this DeepLayer object.
     *
     * @return the layers of this neural network.
     */
    public Iterable<ILayer> getLayers() {
        return Arrays.asList(this.layers);
    }

    public void setLayers(ArrayList<ILayer> layers) {
        this.layers = new ILayer[layers.size()];
        layers.toArray(this.layers);
        checkLayers();
    }

    /**
     * Sets the learning rate for this deep layer.
     *
     * @param lr the learning rate.
     */
    public void setLearningRate(LearningRate lr) {
        this.learningRate = lr;
    }

    /**
     * Returns the learning rate of the deep layer.
     *
     * @return the learning rate.
     */
    public LearningRate getLearningRate() {
        return this.learningRate;
    }

    /**
     * Sets the cost function of this layer. This cost function will also
     * calculate the initial deltas that will be used for back propagation. The
     * cost function should only be used on the final layer.
     *
     * @param function the function to set as cost function.
     */
    public void setCostFunction(CostFunction function) {
        this.costFunction = function;
    }

    /**
     * Returns the cost function of this deep layer.
     *
     * @return the cost function.
     */
    public CostFunction getCostFunction() {
        return costFunction;
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

    public ILayer getLayer(int index) {
        if (index < layers.length && index >= 0) {
            return layers[index];
        } else {
            return null;
        }
    }

    public int getNrOfLayers() {
        return layers.length;
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
     * @param mode
     */
    public void train(int iteration, imatrix input, imatrix target, TrainingMode mode) {
        setTarget(target);
        forward(input);
        float lr = this.learningRate.getLearningRate(iteration);

        ILayer last = getLastLayer();
        this.costFunction.calculateDerivedCost(last.getErrors(), last.getOutputs(), target);
        for (int i = layers.length; i > 0; --i) {
            ILayer current = layers[i - 1];
            // only calculate deltas for last layer.
            current.backpropagate(lr);

            if ((i - 2) >= 0) {
                ILayer previous = layers[i - 2];
                current.calculateErrors(previous.getErrors());
            }
        }

        //if (mode == TrainingMode.STOCHASTIC) {
            for (int i = 0; i < layers.length; ++i) {
                layers[i].adaptWeights(this.learningRate.getLearningRate(iteration));
            }
        //}
    }

    public void adaptWeights(int iteration, int batchSize) {
        float lr = this.learningRate.getLearningRate(iteration);
        for (int i = 0; i < layers.length; ++i) {
            layers[i].adaptWeights(lr / batchSize);
        }
    }

    public void forward(imatrix input) {
        setInputs(input);
        for (int i = 0; i < layers.length; ++i) {
            layers[i].forward();
            if (i + 1 < layers.length) {
                imatrix outputs = layers[i].getOutputs();
                layers[i + 1].setInputs(outputs);
            }
        }
    }

    public void writeOutputImages() {
        int layerIndex = 1;
        Calendar c = Calendar.getInstance();
        int year = c.get(Calendar.YEAR);
        int month = c.get(Calendar.MONTH);
        int day = c.get(Calendar.DAY_OF_MONTH);
        int hour = c.get(Calendar.HOUR_OF_DAY);
        int mins = c.get(Calendar.MINUTE);
        for (ILayer l : this.layers) {
            l.writeOutputImage("output_" + year + "_" + month + "_" + day + "/" + hour + "_" + mins + "_" + layerIndex);
            ++layerIndex;
        }
    }

    public String getTrainingStartTimeAsFolder() {
        Calendar c = Calendar.getInstance();
        int year = c.get(Calendar.YEAR);
        int month = c.get(Calendar.MONTH);
        int day = c.get(Calendar.DAY_OF_MONTH);
        int hour = c.get(Calendar.HOUR_OF_DAY);
        int mins = c.get(Calendar.MINUTE);

        return year + "_" + (month + 1) + "_" + day + "/" + hour + "_" + mins;
    }

    /**
     * Writes a weight representation of every weight.
     *
     * @param folder the base folder for the images.
     * @param iteration the training iteration to write.
     */
    public void writeWeightImages(String folder, int iteration) {
        int layerIndex = 1;
        for (ILayer l : this.layers) {
            l.writeWeightImage(folder + "/" + iteration + "/" + layerIndex + "_" + l.getName());
            ++layerIndex;
        }
    }

    public void setMetaData(DeepLayerMetaData dlmd) {
        this.metadata = dlmd;
    }

    public DeepLayerMetaData getMetaData() {
        return metadata;
    }

    public void analyzeWeights() {
        for (ILayer l : this.layers) {
            l.analyzeWeights();
        }
    }

    public void sync() {
        for (ILayer l : this.layers) {
            l.sync();
        }
    }

}
