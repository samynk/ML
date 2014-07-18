/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer;

import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Koen
 */
public class NeuralNetCell {

    private ActivationFunction function;
    // the previous weights of this neural net cell
    private float[] previousWeights;
    // the weights that are used to calculate the output of this neural net cell.
    private float[] weights;
    // checks if the weights are mutable or not.
    private boolean[] mutable;
    private float minimumInput;
    private float maximumInput;
    /**
     * The min weight for outgoing connections.
     */
    private float minWeight = Float.NEGATIVE_INFINITY;
    /**
     * The max weight for outgoing connections.
     */
    private float maxWeight = Float.POSITIVE_INFINITY;
    /**
     * The layer this neural net cell is a part of.
     */
    private NeuralNetworkLayer parentLayer;
    /**
     * The input layer for this neural net cell.
     */
    private NeuralNetworkLayer inputLayer;
    /**
     * The output layer for this neural net cell.
     */
    private NeuralNetworkLayer outputLayer;
    // the current output ( h(sum of inputs))
    // initialized at one for easier bias setting.
    private float output = 1.0f;
    private boolean biasTerm = false;
    // the current delta (h'(output) (yk - tk)
    private float delta;
    private String name;
    private ArrayList<String> aliases = new ArrayList<>();

    public int getNrOfWeights() {
        if (weights != null) {
            return weights.length;
        } else {
            return 0;
        }
    }

    /**
     * Creates a new NeuralNetCell.
     * @param function
     * @param index 
     */
    public NeuralNetCell(ActivationFunction function) {
        this.function = function;
    }

    public NeuralNetCell(ActivationFunction function, int nrOfInputs) {
        this.function = function;
        this.weights = new float[nrOfInputs];
        this.mutable = new boolean[nrOfInputs];
        this.previousWeights = new float[nrOfInputs];

        for (int i = 0; i < nrOfInputs; ++i) {
            mutable[i] = true;
        }
    }

    public void setAsBiasTerm() {
        this.biasTerm = true;
    }

    public void SetInputLayer(NeuralNetworkLayer input) {
        this.inputLayer = input;
        if (this.weights == null);
        {
            weights = new float[input.getNrOfCells()];
            previousWeights = new float[input.getNrOfCells()];
            mutable = new boolean[input.getNrOfCells()];

            for (int i = 0; i < input.getNrOfCells(); ++i) {
                mutable[i] = true;
            }
        }

        if (this.weights != null) {
            float[] newWeights = new float[input.getNrOfCells()];
            System.arraycopy(weights, 0, newWeights, 0, weights.length);
            this.weights = newWeights;

            previousWeights = new float[input.getNrOfCells()];

            boolean newMutable[] = new boolean[input.getNrOfCells()];

            for (int i = 0; i < input.getNrOfCells(); ++i) {
                newMutable[i] = true;
            }
            System.arraycopy(mutable, 0, newMutable, 0, mutable.length);
            mutable = newMutable;
        }
    }

    public void addAlias(String alias) {
        aliases.add(alias);
        parentLayer.addAlias(alias, this);
    }

    public void setInputWeightMutable(int index, boolean inputMutable) {
        mutable[index] = inputMutable;
    }
    
    public void setAllInputWeightsMutable(boolean inputMutable){
        for (int i = 0 ;  i< mutable.length; ++i)
            mutable[i]=inputMutable;
    }

    /**
     * Just for completeness sake.
     * @param output 
     */
    public void SetOutputLayer(NeuralNetworkLayer output) {
        this.outputLayer = output;
    }

    public void setOutput(float output) {
        this.output = output;
    }

    public void initializeWeights(Random r, float minWeight, float maxWeight) {
        float diff = (maxWeight - minWeight);
        for (int i = 0; i < weights.length; ++i) {
            weights[i] = (r.nextFloat() * diff) + minWeight;
        }
    }

    public void calculateOuput() {
        if (!biasTerm) {
            float sum = 0.0f;
            for (int i = 0; i < weights.length; ++i) {
                NeuralNetCell inputCell = inputLayer.getCellAt(i);
                sum += inputCell.getOutputValue() * weights[i];

            }
            switch (function) {
                case SOFTMAX: {
                    output = (float) Math.exp(sum);
                    break;
                }
                case IDENTITY: {
                    output = sum;
                    break;
                }
                case TANH: {
                    output = (float) Math.tanh(sum);
                    break;
                }
                case LINEAR: {
                    output = sum;
                    break;
                }
                case SIGMOID: {
                    output = 1 / (1 + (float) Math.exp(sum));
                    break;
                }
            }
        }
    }

    public float getOutputValue() {
        return output;
    }

    public void scaleOuput(float scale) {
        this.output *= scale;
    }

    public void backpropagate(float value) {
        this.delta = value;
        if (!biasTerm) {
            for (int i = 0; i < weights.length; ++i) {
                NeuralNetCell inputCell = inputLayer.getCellAt(i);
                inputCell.addDeltaValue(value * weights[i]);
            }
        }
    }

    public void addDeltaValue(float dDelta) {
        delta += dDelta;
    }

    public void resetDeltaValue() {
        delta = 0;
    }

    public void adaptWeights(float learningRate) {
        assert (weights.length == previousWeights.length);
        System.arraycopy(weights, 0, previousWeights, 0, weights.length);


        for (int i = 0; i < weights.length; ++i) {
            NeuralNetCell inputCell = inputLayer.getCellAt(i);
            float dw = -learningRate * inputCell.getOutputValue() * this.delta;
            if (mutable[i]) {
                weights[i] += dw;
                if (weights[i] > inputCell.getMaxWeight()) {
                    weights[i] = inputCell.getMaxWeight();
                } else if (weights[i] < inputCell.getMinWeight()) {
                    weights[i] = inputCell.getMinWeight();
                }
            }
        }
    }

    public void backpropagate() {
        if (!biasTerm) {
            switch (function) {
                case IDENTITY:
                case LINEAR:
                case SOFTMAX:
                    this.delta = -this.output * delta;
                    break;
                case SIGMOID:
                    float eoutput = (float) Math.exp(output);
                    float sigm = 1 / (1 + eoutput);
                    this.delta = delta * eoutput * sigm * sigm;
                    break;
                case TANH: {
                    float tanh = (float) Math.tanh(this.output);
                    this.delta = (1 - tanh * tanh) * delta;
                    break;
                }
            }
            for (int i = 0; i < weights.length; ++i) {
                NeuralNetCell inputCell = inputLayer.getCellAt(i);
                inputCell.addDeltaValue(this.delta * weights[i]);
            }
        }
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setWeight(int inputIndex, float a) {
        this.weights[inputIndex] = a;
    }

    public float getWeight(int inputIndex) {
        return weights[inputIndex];
    }

    public float getPreviousWeight(int inputIndex) {
        return previousWeights[inputIndex];
    }

    public boolean isBiasTerm() {
        return this.biasTerm;
    }

    /**
     * @return the minWeight
     */
    public float getMinWeight() {
        return minWeight;
    }

    /**
     * @param minWeight the minWeight to set
     */
    public void setMinWeight(float minWeight) {
        this.minWeight = minWeight;
    }

    /**
     * @return the maxWeight
     */
    public float getMaxWeight() {
        return maxWeight;
    }

    /**
     * @param maxWeight the maxWeight to set
     */
    public void setMaxWeight(float maxWeight) {
        this.maxWeight = maxWeight;
    }

    public float getMinimumInput() {
        return minimumInput;
    }

    public void setMinimumInput(float min) {
        this.minimumInput = min;
    }

    public float getMaximumInput() {
        return maximumInput;
    }

    public void setMaximumInput(float max) {
        this.maximumInput = max;
    }

    /**
     * @return the parentLayer
     */
    public NeuralNetworkLayer getParentLayer() {
        return parentLayer;
    }

    /**
     * @param parentLayer the parentLayer to set
     */
    public void setParentLayer(NeuralNetworkLayer parentLayer) {
        this.parentLayer = parentLayer;
    }
}