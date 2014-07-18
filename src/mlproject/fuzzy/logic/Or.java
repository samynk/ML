/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy.logic;

import java.util.ArrayList;
import mlproject.fuzzy.FuzzySystem;
import mlproject.fuzzy.MemberShip;

/**
 *
 * @author Koen
 */
public class Or extends RulePart{
    private ArrayList<RulePart> operators = new ArrayList<>();
    
    public Or(){
        
    }
    
    public Or(RulePart op1, RulePart op2){
        operators.add(op1);
        operators.add(op2);
    }
    
    public RulePart getOp1(){
        return operators.get(0);
    }
    
    public RulePart getOp2(){
        return operators.get(1);
    }

    public void addOperator(RulePart a) {
        if ( a instanceof Group){
            Group g= (Group) a;
            operators.add(g.getChild());
            g.setChild(this);
        }else
            operators.add(a);
    }
    
    public float evaluateAntecedent(FuzzySystem system){
        float maximum  = Float.MIN_VALUE;
        for (RulePart rp : operators){
            float result = rp.evaluateAntecedent(system);
            if ( result > maximum ){
                maximum = result;
            }
        }
        return maximum;
    }
    
    public float evaluateTestValue(FuzzySystem system){
        float maximum  = Float.MIN_VALUE;
        for (RulePart rp : operators){
            float result = rp.evaluateTestValue(system);
            if ( result > maximum ){
                maximum = result;
            }
        }
        return maximum;
    }
    
     public void getInputs(FuzzySystem system,ArrayList<MemberShip> inputs){
         for ( RulePart rp : this.operators)
             rp.getInputs(system,inputs);
    }
}
