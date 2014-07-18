/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 *
 * @author Koen
 */
public class FuzzySystem {

    private String name;
    private HashMap<String, FuzzyVariable> fuzzyInputs;
    private HashMap<String, FuzzyVariable> fuzzyOutputs;
    private ArrayList<FuzzyRule> fuzzyRules;
    private FuzzyRuleBlock[] fuzzyRuleBlocks;
    private ArrayList<ChangeListener> ruleChangeListeners = new ArrayList<>();
    private boolean iterativeMode = false;
    private int maxIterations = 20;

    public FuzzySystem(String name) {
        this.name = name;
        fuzzyInputs = new HashMap<>();
        fuzzyOutputs = new HashMap<>();
        fuzzyRules = new ArrayList<>();
        fuzzyRuleBlocks = new FuzzyRuleBlock[1];
    }

    public void addChangeListener(ChangeListener listener) {
        this.ruleChangeListeners.add(listener);
    }

    public void removeChangeListener(ChangeListener listener) {
        this.ruleChangeListeners.remove(listener);
    }

    public void addFuzzyInput(FuzzyVariable input) {
        fuzzyInputs.put(input.getName().toLowerCase(), input);
    }

    public void addFuzzyOutput(FuzzyVariable output) {
        fuzzyOutputs.put(output.getName().toLowerCase(), output);
    }

    public FuzzyVariable getFuzzyInputVariable(String variable) {
        return fuzzyInputs.get(variable.toLowerCase());
    }

    public FuzzyVariable getFuzzyOutputVariable(String variable) {
        return fuzzyOutputs.get(variable.toLowerCase());
    }

    public void addFuzzyRule(FuzzyRule rule) {
        this.addFuzzyRule(rule, 1);
    }

    public void addFuzzyRule(FuzzyRule rule, int rulegroup) {
        rule.setParentSystem(this);
        fuzzyRules.add(rule);
        
        // rule group is one based
        if ( rulegroup > fuzzyRuleBlocks.length ){
            FuzzyRuleBlock[] newArray = new FuzzyRuleBlock[rulegroup];
            System.arraycopy(this.fuzzyRuleBlocks, 0, newArray, 0, fuzzyRuleBlocks.length);
            fuzzyRuleBlocks = newArray;
        }
        if ( fuzzyRuleBlocks[rulegroup-1] == null )
        {
            FuzzyRuleBlock block = new FuzzyRuleBlock(this,rulegroup);
            fuzzyRuleBlocks[rulegroup-1] = block;
        }
        fuzzyRuleBlocks[rulegroup-1].addFuzzyRule(rule);
    }
    
    public FuzzyRuleBlock[] getFuzzyRuleBlocks(){
        return fuzzyRuleBlocks;
    }
    
    public void addFuzzyRule(String rule) {
        this.addFuzzyRule(rule,1);
    }

    public void addFuzzyRule(String rule, int rulegroup) {
        FuzzyRule fuzzyrule = new FuzzyRule(rule);
        addFuzzyRule(fuzzyrule,rulegroup);
    }

    /**
     * Returns the name of the fuzzy system
     *
     * @return the name of the fuzzy system.
     */
    public String getName() {
        return name;
    }

    public void setFuzzyInput(String varName, float value) {
        FuzzyVariable input = fuzzyInputs.get(varName);
        if (input != null) {
            input.setCurrentValue(value);
        }
        //else
        //    System.out.println("Fuzzy input not found : " + varName);
    }

    public void evaluate() {
        for (FuzzyVariable output : fuzzyOutputs.values()) {
            output.resetOutput();
        }
        for (FuzzyRule rule : fuzzyRules) {
            rule.evaluate(this);
        }

    }

    public void evaluate(boolean notify) {

        for (FuzzyVariable output : fuzzyOutputs.values()) {
            output.resetOutput();
        }
        for (FuzzyRule rule : fuzzyRules) {
            rule.evaluate(this);
        }
        if (notify) {
            ChangeEvent ce = new ChangeEvent(this);
            for (ChangeListener cl : ruleChangeListeners) {
                cl.stateChanged(ce);
            }
        }
    }

    public float getFuzzyOutput(String varName) {
        FuzzyVariable output = fuzzyOutputs.get(varName.toLowerCase());
        return output.getOutputValue();
    }

    public Iterable<FuzzyVariable> getInputs() {
        return fuzzyInputs.values();
    }

    public Iterable<FuzzyVariable> getOutputs() {
        return fuzzyOutputs.values();
    }

    public Iterable<FuzzyRule> getRules() {
        return fuzzyRules;
    }

    public Object[] getRulesAsArray() {
        return fuzzyRules.toArray();
    }

    public int getNrOfInputs() {
        return fuzzyInputs.size();
    }

    public int getNrOfOutputs() {
        return fuzzyOutputs.size();
    }

    public void sortRules() {
        Collections.sort(fuzzyRules);
    }

    public int getNrOfRules() {
        return fuzzyRules.size();
    }

    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return the iterativeMode
     */
    public boolean isIterativeMode() {
        return iterativeMode;
    }

    /**
     * @param iterativeMode the iterativeMode to set
     */
    public void setIterativeMode(boolean iterativeMode) {
        this.iterativeMode = iterativeMode;
    }

    public int getMaxIterations() {
        return maxIterations;
    }
}
