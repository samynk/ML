/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject;

import intersect.data.Cell;
import intersect.data.fmatrix;

/**
 *
 * @author Koen
 */
public class SingleLayerNeuralNetwork {
    private fmatrix weightMatrix;
    private fmatrix gradientMatrix;
    private fmatrix inputMatrix;
    
    private fmatrix forwardPassMatrix;
    
    private float learningRate;
    /**
     * 
     * @param nrOfInputs the number of inputs without the bias.
     * @param nrOfOutputs the number of outputs.
     */
    public SingleLayerNeuralNetwork(int nrOfInputs, int nrOfOutputs, float learningRate){
        weightMatrix = fmatrix.random(nrOfOutputs, nrOfInputs+1, -0.03f, +0.03f);
        gradientMatrix = new fmatrix(nrOfOutputs,nrOfInputs+1 );
        inputMatrix = new fmatrix(nrOfInputs+1,1);
        forwardPassMatrix = new fmatrix(nrOfOutputs,1);
        
        this.learningRate = learningRate;
    }
    
    /**
     * The input is a column oriented matrix (every row is an input).
     * @param input the input for the neural network.
     * @param expectedOutput the expected output of the neural network.
     */
    public void doSinglePass(fmatrix input, fmatrix expectedOutput){
        //int maxRow = input.getNrOfRows() < inputMatrix.getNrOfRows()? input.getNrOfRows():inputMatrix.getNrOfRows();
        //for (int row = 1; row <= maxRow; ++row )
        //    inputMatrix.set(row,1,input.get(row,1));
        // set bias
        fmatrix.copyInto(input, inputMatrix);
        inputMatrix.multiply(1/255.0f);
        inputMatrix.set(inputMatrix.getNrOfRows(),1,1.0f);
        
        fmatrix.multiply(forwardPassMatrix,  weightMatrix,inputMatrix);
        
        /*
        forwardPassMatrix.exp();
        float sum = forwardPassMatrix.sum();
        forwardPassMatrix.multiply( 1/sum);
        */
        forwardPassMatrix.softMaxPerColumn();
        
        fmatrix delta = fmatrix.dotsubtract(forwardPassMatrix, expectedOutput);
        
        inputMatrix.transpose();
        //System.out.println("calculating gradient matrix");
        fmatrix.multiply(gradientMatrix,delta,inputMatrix);
        gradientMatrix.multiply( -learningRate);
                
        weightMatrix.add(gradientMatrix);
        
        inputMatrix.transpose();
    }
    
    public boolean predict(fmatrix input, fmatrix expectedOutput){
        int maxRow = input.getNrOfRows() < inputMatrix.getNrOfRows()? input.getNrOfRows():inputMatrix.getNrOfRows();
        for (int row = 1; row <= maxRow; ++row )
            inputMatrix.set(row,1,input.get(row,1));
        // set bias
        inputMatrix.multiply(1/255.0f);
        inputMatrix.set(inputMatrix.getNrOfRows(),1,1.0f);
        
        fmatrix.multiply(forwardPassMatrix,  weightMatrix,inputMatrix);
        
        forwardPassMatrix.exp();
        float sum = forwardPassMatrix.sum();
        forwardPassMatrix.multiply( 1/sum);
        
        // max value is taken as prediction
        Cell check = forwardPassMatrix.max();
        return expectedOutput.get(check.row, 1) > 0;
           
    }

    public int predict(fmatrix input) {
        int maxRow = input.getNrOfRows() < inputMatrix.getNrOfRows()? input.getNrOfRows():inputMatrix.getNrOfRows();
        for (int row = 1; row <= maxRow; ++row )
            inputMatrix.set(row,1,input.get(row,1));
        // set bias
        inputMatrix.multiply(1/255.0f);
        inputMatrix.set(inputMatrix.getNrOfRows(),1,1.0f);
        
        fmatrix.multiply(forwardPassMatrix,  weightMatrix,inputMatrix);
        
        forwardPassMatrix.exp();
        float sum = forwardPassMatrix.sum();
        forwardPassMatrix.multiply( 1/sum);
        
        // max value is taken as prediction
        Cell check = forwardPassMatrix.max();
        return check.row-1;
    }
}
