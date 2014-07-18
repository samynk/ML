/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer;

import java.util.ArrayList;

/**
 *
 * @author Koen
 */
public class SigmoidMemberShipValidator implements WeightValidator{
    /**
     * The weight index for the incoming a-parameter.
     */
    private int aIndex;
    /**
     * The weight index for the incoming ab-parameter.
     */
    private int abIndex;
    
    public SigmoidMemberShipValidator(int aIndex, int abIndex){
        this.aIndex = aIndex;
        this.abIndex = abIndex;
    }
    
    @Override
    public void validate(ArrayList<NeuralNetCell> cells) {
        System.out.println("Validating");
        System.out.println("----------");
        for (NeuralNetCell nnc : cells){
            float aWeight = nnc.getWeight(aIndex);
            float abWeight = nnc.getWeight(abIndex);
            
            float b = abWeight;
            System.out.println(nnc.getName()+ "b:" + b);
        }
        
    }
    
}
