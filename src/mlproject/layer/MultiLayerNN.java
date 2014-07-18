/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer;

import intersect.data.fmatrix;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Koen
 */
public class MultiLayerNN {
    private ArrayList<NeuralNetworkLayer> layers = new ArrayList<>();
    
    public void MultiLayerNN(){
        
    }
    
    public void addLayer(NeuralNetworkLayer nnl ){
        if (layers.size() > 0 ){
            NeuralNetworkLayer last = layers.get(layers.size()-1);
            nnl.setInputLayer(last);
            last.setOutputLayer(nnl);
        }
        layers.add(nnl);
    }
    
     /**
     * Sets the input values for this single layer neural network.
     * @param values the values for this single layer neural network.
     */
    public void setInputs(fmatrix values){
        layers.get(0).setOutputs(values);
    }
    
    /**
     * Initializes the weight values for the cells of the neural network
     */
    public void initialize(float minWeight, float maxWeight){
        Random r = new Random(System.currentTimeMillis());
        for (int i = 1 ; i < layers.size();++i)
            layers.get(i).initializeWeights(r,minWeight,maxWeight); 
    }
    
    public NeuralNetworkLayer getInputLayer(){
        return layers.get(0);
    }
    
    public NeuralNetworkLayer getOutputLayer(){
        return layers.get(layers.size()-1);
    }
    
    /**
     * Do the forward calculation of this 
     * In this case of classification the index of the cell with the biggest
     * output value is returned.
     */
    public int forward(fmatrix inputs){
        getInputLayer().setOutputs(inputs);
        for(int i = 1; i < layers.size();++i)
            layers.get(i).calculateOutput();
        return getOutputLayer().getArgMax();
    }
    
    /**
     * Adapts the weights, based on the expected output.
     * @param expectedOutput the expected output of the neural network.
     * @param learningRate the learning rate for the back propagation.
     */
    public void backpropagate(float learningRate, fmatrix expectedOutput){
        for (NeuralNetworkLayer nnl : layers){
            nnl.resetDeltaValues();
        }
        getOutputLayer().backpropagate(expectedOutput);
        for (int i = layers.size()-2;i > 0; --i){
            // backpropagate on the basis of the temporary deltas.
            layers.get(i).backpropagate();
        }
        
        for (int i = layers.size()-1; i > 0; --i){
            NeuralNetworkLayer layer = layers.get(i);
            if ( layer.isMutable())
                layers.get(i).adaptWeights(learningRate);
        }
    }
    
    public void printLayer(int layerIndex){
        if ( layerIndex >= 0 && layerIndex < layers.size())
        {
            NeuralNetworkLayer layer = layers.get(layerIndex);
            for (int i = 0; i < layer.getNrOfCells(); ++i)
            {
                System.out.println("value " + i + ":" + layer.getCellAt(i).getOutputValue());
            }
        }
    }

    public int getNrOfLayers() {
        return this.layers.size();
    }

    public NeuralNetworkLayer getLayer(int i) {
        return layers.get(i);
    }

    public void setInput(String inputName, float f) {
        NeuralNetCell cell  = this.layers.get(0).getCellWithName(inputName);
        if ( cell != null)
            cell.setOutput(f);
    }

    public void forward() {
       for(int i = 1; i < layers.size();++i)
            layers.get(i).calculateOutput();
    }
    
    public float getOutput(){
        return getOutputLayer().getOutput(0);
    }
}
