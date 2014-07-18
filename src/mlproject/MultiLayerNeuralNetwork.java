/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject;

import intersect.data.Cell;
import intersect.data.fmatrix;
import java.util.ArrayList;

/**
 * Class that describes a multilayer neural network
 * @author Koen
 */
public class MultiLayerNeuralNetwork {
    private int nrOfInputs;
    private int nrOfOutputs;
    
    private int[] hiddenLayerSizes;
    
    private float learningRate = 0.01f;
    
    private fmatrix firstInput;
    
    private ArrayList<fmatrix> weightMatrices = new ArrayList<fmatrix>();
    private ArrayList<fmatrix> gradientMatrices = new ArrayList<fmatrix>();
    private ArrayList<fmatrix> layerOutputs = new ArrayList<fmatrix>();
    private ArrayList<fmatrix> deltas = new ArrayList<fmatrix>();
    private ArrayList<fmatrix> deltasWithoutBias = new ArrayList<fmatrix>();
    
    /**
     * Creates a new multi layer neural network with an arbitrary number of hidden layers
     * and units in these hidden layers. Error back propagation is used to train the network.
     * The bias units are added automatically for each layer.
     * The outputs of the layers are column oriented.
     * @param nrOfInputs the number of inputs.
     * @param nrOfOutput the number of output.
     * @param hiddenLayerSize the size of the hidden layer.
     */
    public MultiLayerNeuralNetwork(int nrOfInputs, int nrOfOutputs, int ... hiddenLayerSizes ){
        this.nrOfInputs = nrOfInputs;
        this.nrOfOutputs = nrOfOutputs;
        // add bias to firstInput;
        firstInput = new fmatrix(nrOfInputs+1,1);
        
        
        this.hiddenLayerSizes = hiddenLayerSizes;
        if ( hiddenLayerSizes.length == 0 ){
            fmatrix first = fmatrix.random(nrOfInputs+1,nrOfOutputs,-0.003f,0.003f);
            weightMatrices.add(first);
            fmatrix firstGradient = new fmatrix(nrOfInputs+1,nrOfOutputs);
            gradientMatrices.add(firstGradient);
            
            fmatrix output = new fmatrix(nrOfOutputs,1);
            layerOutputs.add(output);
            fmatrix delta = new fmatrix(nrOfOutputs,1);
            deltas.add(delta);     
        }else{
            
            // first weightmatrix : connects nrOfInputs and first hidden layer
            fmatrix first = fmatrix.random( hiddenLayerSizes[0],nrOfInputs+1,-1f,1f);
            weightMatrices.add(first);
            
            fmatrix firstGradient = new fmatrix( hiddenLayerSizes[0],nrOfInputs+1);
            gradientMatrices.add(firstGradient);
            
            fmatrix firstOutput =new fmatrix(hiddenLayerSizes[0]+1,1);
            layerOutputs.add(firstOutput);
            
            fmatrix firstDelta = new fmatrix(hiddenLayerSizes[0]+1,1);
            deltas.add(firstDelta);
            
            fmatrix firstDeltaWithoutBias = new fmatrix(hiddenLayerSizes[0],1);
            deltasWithoutBias.add(firstDeltaWithoutBias);
            
            for (int i = 0; i < hiddenLayerSizes.length - 1; ++i){
                fmatrix hidden= fmatrix.random(hiddenLayerSizes[i+1],hiddenLayerSizes[i]+1, -1f, 1f);
                weightMatrices.add(hidden);
                
                fmatrix hiddenGradient= new fmatrix(hiddenLayerSizes[i+1],hiddenLayerSizes[i]+1);
                gradientMatrices.add(hiddenGradient);
                
                fmatrix output = new fmatrix(hiddenLayerSizes[i+1]+1,1);
                layerOutputs.add(output);
                fmatrix delta = new fmatrix(hiddenLayerSizes[i+1]+1,1);
                deltas.add(delta);
                
                fmatrix deltaWithoutBias = new fmatrix(hiddenLayerSizes[i+1],1);
                deltasWithoutBias.add(deltaWithoutBias);
            }    
            
            int nrOfLastInputs = hiddenLayerSizes[hiddenLayerSizes.length-1];
            fmatrix last = fmatrix.random( nrOfOutputs, nrOfLastInputs+1,-1f,1f);
            weightMatrices.add(last);
            
            fmatrix lastGradient =  new fmatrix( nrOfOutputs, nrOfLastInputs+1);
            gradientMatrices.add(lastGradient);
            
            fmatrix output = new fmatrix(nrOfOutputs,1);
            layerOutputs.add(output);
            fmatrix delta = new fmatrix(nrOfOutputs,1);
            deltas.add(delta);
        }
    }
    
     /**
     * The input is a column oriented matrix (every row is an input).
     * The deltas are calculated and the back propagation is performed.
     * @param input the input for the neural network.
     * @param expectedOutput the expected output of the neural network.
     */
    public void doSinglePass(fmatrix input, fmatrix expectedOutput){
        fmatrix.copyInto(input, firstInput);
        // set bias element
        firstInput.set(nrOfInputs+1,1,1.0f);
        firstInput.multiply(1/255.0f);
        
        fmatrix currentInput = firstInput;
        for(int layer = 0; layer < weightMatrices.size(); ++layer)
        {
            fmatrix currentOutput = layerOutputs.get(layer);
            fmatrix.multiply(currentOutput, weightMatrices.get(layer), currentInput);
            
            if ( layer+1 < weightMatrices.size())
            {
                // apply tangens hyperbolicus.
                
                currentOutput.tanh();
                 currentOutput.set(currentOutput.getNrOfRows(),1,1);
               
            }else{
                // last layer softmax
                currentOutput.softMaxPerColumn();
            }
            // set bias element
            
            currentInput = currentOutput;
        }
        
        // calculate deltas and update weights
        fmatrix lastDelta = deltas.get(deltas.size()-1);
        fmatrix.dotsubtract(lastDelta,expectedOutput, currentInput);
        fmatrix currentDelta = lastDelta;
        for(int layer = weightMatrices.size()-1;layer >=0; --layer)
        {
            fmatrix weightMatrix = weightMatrices.get(layer);
            if (layer > 0 ){
                fmatrix nextDelta = deltas.get(layer-1);
                weightMatrix.transpose();
                fmatrix hiddenOutput = this.layerOutputs.get(layer-1);
                hiddenOutput.difftanh();
                fmatrix.dotmultiply(hiddenOutput, hiddenOutput,nextDelta);
                //fmatrix.dothiddenOutput.
                fmatrix.multiply(currentDelta,weightMatrix,hiddenOutput);
                
                //nextDelta.difftanh();
                currentDelta = nextDelta;
                weightMatrix.transpose();
            }
        }
        
        currentInput = firstInput;
        for(int layer = 0;layer < weightMatrices.size();++layer){
            fmatrix weightMatrix = weightMatrices.get(layer);
            fmatrix gradientMatrix = gradientMatrices.get(layer);
            fmatrix delta = deltas.get(layer);
            currentInput.transpose();
            
            if ( delta.getNrOfRows() > gradientMatrix.getNrOfRows()){
                // remove the bias
                fmatrix deltaWithoutBias = deltasWithoutBias.get(layer);
                fmatrix.copyInto(delta, deltaWithoutBias);
                delta = deltaWithoutBias;
            }
            
            fmatrix.multiply(gradientMatrix,delta,currentInput);
            
            if ( layer < weightMatrices.size()-1){
                gradientMatrix.difftanh(); 
            }
            gradientMatrix.multiply(learningRate);
            weightMatrix.add(gradientMatrix);
            currentInput.transpose();
            //weightMatrix.clamp(-2f,2f);
            currentInput = this.layerOutputs.get(layer);
        }
    }
    
    public static void main(String[] args) {
        MultiLayerNeuralNetwork mlnn = new MultiLayerNeuralNetwork(784,10,40);
        
    }

    public boolean predict(fmatrix column, fmatrix expected) {
        fmatrix.copyInto(column, firstInput);
        // set bias element
        
        firstInput.multiply(1/255.0f);
        firstInput.set(nrOfInputs+1,1,1.0f);
        
        fmatrix currentInput = firstInput;
        for(int layer = 0; layer < weightMatrices.size(); ++layer)
        {
            fmatrix currentOutput = layerOutputs.get(layer);
            fmatrix.multiply(currentOutput, weightMatrices.get(layer), currentInput);
            
            if ( layer+1 < weightMatrices.size())
            {
                // apply tangens hyperbolicus.
                
                currentOutput.tanh();
                currentOutput.set(currentOutput.getNrOfRows(),1,1);
                 
            }else{
                // last layer softmax
                currentOutput.softMaxPerColumn();
            }
            // set bias element
           
            currentInput = currentOutput;
        }
        Cell check = currentInput.max();
        return expected.get(check.row, 1) > 0;
    }
}
