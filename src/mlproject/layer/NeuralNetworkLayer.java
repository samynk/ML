/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer;

import intersect.data.fmatrix;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 *
 * @author Koen
 */
public class NeuralNetworkLayer {
    private ArrayList<NeuralNetCell> cells;
    private ActivationFunction function;
    
    private boolean biasTerm = false;
    
    private NeuralNetworkLayer inputLayer;
    
    private String name;
    
    private HashMap<String,ArrayList<NeuralNetCell>> subGroups =
            new HashMap<>();
    
    private HashMap<String,WeightValidator> subGroupValidator = 
            new HashMap<>();
    
    private HashMap<String,NeuralNetCell> aliasMap = 
            new HashMap<>();
    
    /**
     * Controls the mutability of the weights in this layer.
     */
    private boolean mutable;
    
    public NeuralNetworkLayer(boolean biasTerm,ActivationFunction function){
        this.function = function;
        cells = new ArrayList<>();
        
        this.biasTerm = biasTerm;
        if  (biasTerm){
            NeuralNetCell biasCell = new NeuralNetCell(function);
            biasCell.setParentLayer(this);
            biasCell.setAsBiasTerm();
            cells.add(biasCell);
        }
    }
    
    public NeuralNetworkLayer(int nrOfCells, boolean biasTerm,ActivationFunction function){
        this.function = function;
        cells = new ArrayList<>(nrOfCells);
        for (int i = 0 ; i < nrOfCells; ++i){
            NeuralNetCell nnc = new NeuralNetCell(function);
            nnc.setParentLayer(this);
            cells.add(nnc);
        }
        this.biasTerm = biasTerm;
        if  (biasTerm){
            NeuralNetCell biasCell = cells.get(cells.size()-1);
            biasCell.setParentLayer(this);
            biasCell.setAsBiasTerm();
        }
    }
    
    public void addCellToGroup(String group, NeuralNetCell cell){
        if ( cells.contains(cell)){
            ArrayList<NeuralNetCell> subGroup = subGroups.get(group);
            if ( subGroup == null){
                subGroup = new ArrayList<>();
                subGroups.put(group, subGroup);
            }
            if ( !subGroup.contains(cell))
                subGroup.add(cell);
        }
    }
    
    public void validateWeights(){
        for (String key : this.subGroupValidator.keySet())
        {
            ArrayList<NeuralNetCell> subGroup = subGroups.get(key);
            if( subGroup != null){
                WeightValidator wv = subGroupValidator.get(key);
                wv.validate(subGroup);
            }
        }
    }
    
    public String getName(){
        return name;
    }
    
    public void setName(String name){
        this.name = name;
    }
    
    /**
     * Checks if this neural network layer is mutable.
     * @return 
     */
    public boolean isMutable(){
        return mutable;
    }
    
    /**
     * Returns the mutable state of this neural network layer.
     * @param mutable set to true if the layer is mutable, false otherwise.
     */
    public void setMutable(boolean mutable){
        this.mutable = mutable;
    }
    
    /**
     * Sets the outputs of this NeuralNetworklayer manually. (This is a convenience
     * for a layer that is the output layer. The matrix can be row oriented
     * or column oriented.
     * @param values 
     */
    public void setOutputs(fmatrix values){
        if ( values.getNrOfRows()== cells.size()
                || values.getNrOfRows() == cells.size()-1){
            for (int i = 0; i < values.getNrOfRows(); ++i){
                NeuralNetCell nnc = cells.get(i);
                nnc.setOutput(values.get(i+1,1));
            }
        }else if ( values.getNrOfColumns() == cells.size()
                || values.getNrOfColumns() == cells.size()-1){
            for (int i = 0; i < values.getNrOfColumns(); ++i){
                NeuralNetCell nnc = cells.get(i);
                nnc.setOutput(values.get(1,i+1));
            }
        }
    }
    
    public void setInputLayer(NeuralNetworkLayer input){
        if (input != this.inputLayer) {
            this.inputLayer = input;
            for (NeuralNetCell cell : cells) {
                cell.SetInputLayer(input);
            }
        }
    }
    
    public void setOutputLayer(NeuralNetworkLayer output){
        for (NeuralNetCell cell:cells){
            cell.SetOutputLayer(output);
        }
    }
    
    public int getNrOfCells(){
        return cells.size();
    }

    public void initializeWeights(Random r,float minWeight, float maxWeight) {
        for (NeuralNetCell cell: cells){
            cell.initializeWeights(r,minWeight,maxWeight);
        }
    }

    public void calculateOutput() {
        for ( NeuralNetCell cell: cells){
            cell.calculateOuput();
        }
        if ( function == ActivationFunction.SOFTMAX ){
            float outputSum = this.calculateSum();
            scaleOutputs(1/outputSum);
        }
    }
    
    public void scaleOutputs(float scale){
        for ( NeuralNetCell cell: cells){
            cell.scaleOuput(scale);
        }
    }
    
    public int getArgMax(){
        float max = Float.MIN_VALUE;
        int index = -1;
        for (int i = 0; i < cells.size(); ++i){
            NeuralNetCell cell = cells.get(i);
            if (cell.getOutputValue() > max ){
                max = cell.getOutputValue();
                index = i;
            }
        }
        return index;
    }
    
    public float calculateSum(){
        float sum = 0;
        for (NeuralNetCell cell:cells){
            sum += cell.getOutputValue();
        }
        return sum;
    }

    public NeuralNetCell getCellAt(int i) {
        return cells.get(i);
    }
    
    public int getIndexOf(NeuralNetCell cell){
        return cells.indexOf(cell);
    }
    
    public void backpropagate(fmatrix output)
    {
        if (output.getNrOfRows() == cells.size())
        {
            for(int i = 0; i < cells.size(); ++i)
            {
                NeuralNetCell cell = cells.get(i);
                float delta = (cell.getOutputValue() - output.get(i+1, 1));
                cell.backpropagate(delta);
            }
        }else if ( output.getNrOfColumns()==cells.size())
        {
            for(int i = 0; i < cells.size(); ++i)
            {
                NeuralNetCell cell = cells.get(i);
                float delta = cell.getOutputValue() - output.get(1,i+1);
                cell.backpropagate(delta);
            }
        }
    }
    
    public void resetDeltaValues(){
        for (NeuralNetCell nnc:cells){
            nnc.resetDeltaValue();
        }
    }

    /**
     * Use the deltas to adapt the weights.
     */
    public void adaptWeights(float learningRate) {
        for( NeuralNetCell cell : cells)
        {
            cell.adaptWeights(learningRate);
        }
        validateWeights();
    }

    public void backpropagate() {
         for(int i = 0; i < cells.size(); ++i)
        {
            NeuralNetCell cell = cells.get(i);
            cell.backpropagate();
        }
    }

    public void setCellName(int i, String name) {
        if ( i < cells.size() && i >= 0)
        {
            NeuralNetCell cell = cells.get(i);
            cell.setName(name);
        }
    }

    public NeuralNetCell addCell(int nrOfInputs) {
        NeuralNetCell cell = null;
        if ( biasTerm){
            cell = new NeuralNetCell(this.function,nrOfInputs);
            cell.setParentLayer(this);
            cells.add(cells.size()-1, cell);
            
        }else{
            cell = new NeuralNetCell(this.function,nrOfInputs);
            cell.setParentLayer(this);
            cells.add(cell);
        }
        cell.SetInputLayer(this.inputLayer);
        return cell;
    }
    
    public NeuralNetCell addCell() {
        int nrOfInputs = 0;
        if (inputLayer != null) {
            nrOfInputs = inputLayer.getNrOfCells();
        }
        NeuralNetCell cell = null;
        if (biasTerm) {
            cell = new NeuralNetCell(this.function, nrOfInputs);
            cell.setParentLayer(this);
            cells.add(cells.size() - 1, cell);

        } else {
            cell = new NeuralNetCell(this.function, nrOfInputs);
            cell.setParentLayer(this);
            cells.add(cell);
        }
        cell.SetInputLayer(this.inputLayer);
        return cell;
    }
    
    /**
     * Returns a cell with the given name.
     * @param name the cell name
     * @return the NeuralNetCell object that matches the name.
     */
    public NeuralNetCell getCellWithName(String name){
        for ( NeuralNetCell cell:this.cells)
        {
            if (name.equals(cell.getName()))
                return cell;
        }
        return null;
    }

    public int getCellIndexWithName(String name) {
        for(int i = 0; i < cells.size(); ++i){
            NeuralNetCell cell = cells.get(i);
            if (cell.getName().equals(name))
                return i;
        }
        return -1;
    }

    public float getOutput(int i) {
        return cells.get(i).getOutputValue();
    }

    public NeuralNetCell getLastCell() {
        return cells.get(cells.size()-1);
    }

    public void addWeightValidator(String name, SigmoidMemberShipValidator smsv) {
        this.subGroupValidator.put(name, smsv);
    }

    public boolean hasBiasTerm() {
        return this.biasTerm;
    }

    protected void addAlias(String alias, NeuralNetCell cell) {
        this.aliasMap.put(alias,cell);
    }

    protected NeuralNetCell getCellByAlias(String alias){
        return aliasMap.get(alias);
    }
}
