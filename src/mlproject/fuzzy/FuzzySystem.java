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
    private ArrayList<FuzzyVariable> fuzzyInputList;
    private HashMap<String, FuzzyVariable> fuzzyOutputs;
    private ArrayList<FuzzyVariable> fuzzyOutputList;
    private ArrayList<FuzzyRule> fuzzyRules;
    private ArrayList<FuzzyRuleBlock> fuzzyRuleBlocks = new ArrayList<>();
    private ArrayList<ChangeListener> ruleChangeListeners = new ArrayList<>();
    private boolean iterativeMode = false;
    private int maxIterations = 20;

    public FuzzySystem(String name) {
        this.name = name;
        fuzzyInputs = new HashMap<>();
        fuzzyInputList = new ArrayList<>();
        fuzzyOutputs = new HashMap<>();
        fuzzyOutputList = new ArrayList<>();
        fuzzyRules = new ArrayList<>();
        //default rule block.
    }
    
    /**
     * Creates a clone of the fuzzy system.
     * @return a clone of the fuzzy system.
     */
    public FuzzySystem clone(){
        FuzzySystem clone = new FuzzySystem(name);
        for( FuzzyVariable fv : this.getInputs())
        {
            FuzzyVariable varClone = fv.clone();
            clone.addFuzzyInput(varClone);
        }
        for( FuzzyVariable fv : this.getOutputs())
        {
            FuzzyVariable varClone = fv.clone();
            clone.addFuzzyOutput(varClone);
        }
        for ( FuzzyRuleBlock block : getFuzzyRuleBlocks())
        {
            FuzzyRuleBlock blockClone = new FuzzyRuleBlock(clone,block.getName());
            clone.addFuzzyRuleBlock(blockClone);
            for ( FuzzyRule rule: block.getRules())
            {
                FuzzyRule ruleClone = new FuzzyRule(rule.getRuleText());
                blockClone.addFuzzyRule(ruleClone);
            }
        }
        return clone;
    }
    

    public void addChangeListener(ChangeListener listener) {
        this.ruleChangeListeners.add(listener);
    }

    public void removeChangeListener(ChangeListener listener) {
        this.ruleChangeListeners.remove(listener);
    }

    public void addFuzzyInput(FuzzyVariable input) {
        fuzzyInputs.put(input.getName().toLowerCase(), input);
        fuzzyInputList.add(input);
    }

    public void removeFuzzyInput(FuzzyVariable input) {
        fuzzyInputs.remove(input.getName());
        fuzzyInputList.remove(input);
    }

    public FuzzyVariable getFuzzyInputAt(int index) {
        return fuzzyInputList.get(index);
    }

    public void addFuzzyOutput(FuzzyVariable output) {
        fuzzyOutputs.put(output.getName().toLowerCase(), output);
        fuzzyOutputList.add(output);
    }

    public void removeFuzzyOutput(FuzzyVariable output) {
        fuzzyOutputs.remove(output.getName());
        fuzzyOutputList.remove(output);
    }

    public FuzzyVariable getFuzzyOutputAt(int index) {
        return fuzzyOutputList.get(index);
    }

    public FuzzyVariable getFuzzyInputVariable(String variable) {
        return fuzzyInputs.get(variable.toLowerCase());
    }

    public boolean hasFuzzyInputVariable(String variable) {
        return fuzzyInputs.containsKey(variable);
    }

    public int getInputIndex(FuzzyVariable fv) {
        return fuzzyInputList.indexOf(fv);
    }

    public FuzzyVariable getFuzzyOutputVariable(String variable) {
        return fuzzyOutputs.get(variable.toLowerCase());
    }
    
    public boolean hasFuzzyOutputVariable(String variable) {
        return fuzzyOutputs.containsKey(variable);
    }

    public void addFuzzyRule(FuzzyRule rule) {
        this.addFuzzyRule(rule, "default");
    }

    public void addFuzzyRule(FuzzyRule rule, String rulegroup) {
        rule.setParentSystem(this);
        fuzzyRules.add(rule);


    }

    public Iterable<FuzzyRuleBlock> getFuzzyRuleBlocks() {
        return fuzzyRuleBlocks;
    }

    public void addFuzzyRuleBlock(FuzzyRuleBlock ruleBlock) {
        fuzzyRuleBlocks.add(ruleBlock);
    }

    public void addFuzzyRule(String rule) {
        this.addFuzzyRule(rule, 1);
    }

    public void addFuzzyRule(String rule, int rulegroup) {
        FuzzyRule fuzzyrule = new FuzzyRule(rule);
        addFuzzyRule(fuzzyrule, "ruleblock" + rulegroup);
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

    public int getNrOfRuleBlocks() {
        return this.fuzzyRuleBlocks.size();
    }

    public FuzzyRuleBlock getFuzzyRuleBlock(int index) {
        return fuzzyRuleBlocks.get(index);
    }

    public int getIndexOfFuzzyRuleBlock(FuzzyRuleBlock rb) {
        return fuzzyRuleBlocks.indexOf(rb);
    }

    public int getOutputIndex(FuzzyVariable fv) {
        return fuzzyOutputList.indexOf(fv);
    }

    public boolean hasRuleBlock(String blockName) {
        for( FuzzyRuleBlock block : this.fuzzyRuleBlocks)
        {
            if ( block.getName().equals(blockName))
                return true;
        }
        return false;
    }
}
