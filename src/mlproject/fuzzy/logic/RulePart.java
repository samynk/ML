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
public class RulePart {
    public float evaluateAntecedent(FuzzySystem system){
        return 0.0f;
    }
    
    public float evaluateTestValue(FuzzySystem system){
        return 0.0f;
    }
    
    public void getInputs(FuzzySystem system,ArrayList<MemberShip> inputs){
        
    }

    public float evaluateAntecedentVerbally(FuzzySystem system) {
        return 0.0f;
    }
}
