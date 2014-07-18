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
public class Group extends RulePart{
    private RulePart child;
    
    public void setChild(RulePart part){
        this.child = part;
    }
    
    @Override
    public float evaluateAntecedent(FuzzySystem system){
        return child.evaluateAntecedent(system);
    }
    
    @Override
    public float evaluateAntecedentVerbally(FuzzySystem system){
        return child.evaluateAntecedent(system);
    }
    
    @Override
    public float evaluateTestValue(FuzzySystem system){
        return child.evaluateTestValue(system);
    }
    
    @Override
    public void getInputs(FuzzySystem system, ArrayList<MemberShip> inputs){
        child.getInputs(system,inputs);
    }

    public RulePart getChild() {
        return child;
    }
}
