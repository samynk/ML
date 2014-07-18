/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer;

import intersect.data.fmatrix;
import java.util.Random;

/**
 *
 * @author Koen
 */
public class SingleLayerNN {

    private NeuralNetworkLayer input;
    private NeuralNetworkLayer output;

    public SingleLayerNN(int inputs, int outputs, ActivationFunction function) {
        input = new NeuralNetworkLayer(inputs, true, ActivationFunction.IDENTITY);
        output = new NeuralNetworkLayer(outputs, false, function);

        output.setInputLayer(input);
    }

    /**
     * Sets the input values for this single layer neural network.
     * @param values the values for this single layer neural network.
     */
    public void setInputs(fmatrix values) {
        input.setOutputs(values);
    }

    /**
     * Initializes the weight values for the cells of the neural network
     */
    public void initialize(float minWeight, float maxWeight) {
        Random r = new Random(System.currentTimeMillis());
        output.initializeWeights(r, minWeight, maxWeight);

    }

    /**
     * Do the forward calculation of this 
     * In this case of classification the index of the cell with the biggest
     * output value is returned.
     */
    public int forward(fmatrix inputs) {
        input.setOutputs(inputs);
        output.calculateOutput();
        return output.getArgMax();
    }

    /**
     * Adapts the weights, based on the expected output.
     * @param expectedOutput the expected output of the neural network.
     * @param learningRate the learning rate for the back propagation.
     */
    public void backpropagate(float learningRate, fmatrix expectedOutput) {
        input.resetDeltaValues();
        output.resetDeltaValues();

        output.backpropagate(expectedOutput);
        output.adaptWeights(learningRate);
    }

    public void printOutputLayer() {

        NeuralNetworkLayer layer = output;
        for (int i = 0; i < layer.getNrOfCells(); ++i) {
            System.out.println("value " + i + ":" + layer.getCellAt(i).getOutputValue());
        }
    }
    
      public void printInputLayer() {

        NeuralNetworkLayer layer = output;
        for (int i = 0; i < layer.getNrOfCells(); ++i) {
            System.out.println("value " + i + ":" + layer.getCellAt(i).getOutputValue());
        }
    }

   public  float getOutput(int i) {
        return output.getCellAt(i).getOutputValue();
    }

    public  void setInputCellName(int i, String cellName) {
        if ( i >= 0 && i <input.getNrOfCells())
        {
            input.getCellAt(i).setName(cellName);
        }
    }

    public int getNrOfInputCells(){
        return input.getNrOfCells();
    }
    
    public  NeuralNetCell getInputCellAt(int i){
        if ( i >= 0 && i <input.getNrOfCells())
        {
            return input.getCellAt(i);
        }
        return null;
    }
    
    public  NeuralNetCell getOutputCellAt(int i){
        if ( i >= 0 && i <output.getNrOfCells())
        {
            return output.getCellAt(i);
        }
        return null;
    }
}
