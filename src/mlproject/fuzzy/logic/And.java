/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy.logic;

import java.util.ArrayList;
import mlproject.fuzzy.FuzzySystem;
import mlproject.fuzzy.FuzzyVariable;
import mlproject.fuzzy.MemberShip;

/**
 *
 * @author Koen
 */
public class And extends RulePart{
    private ArrayList<RulePart> operators = new ArrayList<>();
    
    public And(){
        
    }
    
    public And(RulePart op1, RulePart op2){
        operators.add(op1);
        operators.add(op2);
    }
    
    public RulePart getOp1(){
        return operators.get(0);
    }
    
    public RulePart getOp2(){
        return operators.get(1);
    }
    
    public void addOperator(RulePart a){
         operators.add(a);
    }
    
    @Override
     public float evaluateAntecedent(FuzzySystem system){
        
        float minimum = Float.MAX_VALUE;
        for (RulePart rp : operators){
            float result = rp.evaluateAntecedent(system);
            if ( result < minimum ){
                minimum = result;
            }
        }
        return minimum;
        
        
         /**
         float product = 1.0f;
         
         for (RulePart rp : operators){
            float result = rp.evaluateAntecedent(system);
            product *= result;
        }
        return product;
          *  */
        /* 
         float sum = 0.0f;
         for (RulePart rp : operators){
            float result = rp.evaluateAntecedent(system);
            sum += result;
        }
        return sum / operators.size();
        */
        // bounded product
        
    }
    
    @Override
    public float evaluateAntecedentVerbally(FuzzySystem system) {
        float minimum = Float.MAX_VALUE;
        for (RulePart rp : operators){
            float result = rp.evaluateAntecedentVerbally(system);
            if ( result < minimum ){
                minimum = result;
            }
        }
        System.out.println("Result of and : " + minimum);
        return minimum;
    }
     
    @Override
    public float evaluateTestValue(FuzzySystem system){
        /* 
          * float minimum = Float.MAX_VALUE;
        for (RulePart rp : operators){
            float result = rp.evaluateAntecedent(system);
            if ( result < minimum ){
                minimum = result;
            }
        }
        return minimum;
        
         */
        
        
        
        if (operators.size() == 2) {
            float a = operators.get(0).evaluateTestValue(system);
            float b = operators.get(1).evaluateTestValue(system);
            
            if ( a+b < 1.0)
                return 0.0f;
            else 
                return (a+b-1);
        } else {
            float product = 1.0f;

            for (RulePart rp : operators) {
                float result = rp.evaluateTestValue(system);
                product *= result;
            }
            return product;
        }
    }
     
    @Override
     public void getInputs(FuzzySystem system, ArrayList<MemberShip> inputs){
         for ( RulePart rp : this.operators)
             rp.getInputs(system,inputs);
    }
}
