/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

import java.util.ArrayList;
import mlproject.fuzzy.logic.Antecedent;

/**
 *
 * @author Koen Samyn
 */
public class FuzzyRuleBlock {

    private FuzzySystem parent;
    private ArrayList<FuzzyRule> rules = new ArrayList<FuzzyRule>();
    private ArrayList<FuzzyVariable> inputs = new ArrayList<>();
    private ArrayList<FuzzyVariable> outputs = new ArrayList<>();
    private String name;

    public FuzzyRuleBlock(FuzzySystem parent, String name) {
        this.name = name;
        this.parent = parent;
    }
    
    public void clear() {
        rules.clear();
        inputs.clear();
        outputs.clear();
    }
    
    public String getName(){
        return name;
    }

    public void addFuzzyRule(FuzzyRule rule) {
        this.rules.add(rule);
        for (Antecedent a : rule.getAntecedents()) {
            String var = a.getVariable();
            FuzzyVariable input = parent.getFuzzyInputVariable(var);
            if (!inputs.contains(input) && input != null) {
                inputs.add(input);
            }
        }
        String ovar = rule.getOutputVariable();
        FuzzyVariable output = parent.getFuzzyOutputVariable(ovar);
        if (!outputs.contains(output) && output != null) {
            outputs.add(output);
        }
    }
    
    public void evaluate(){
        for ( FuzzyVariable output: this.outputs){
            output.resetOutput();
        }
        for (FuzzyRule rule: this.rules){
            rule.evaluate(parent);
        }
    }

    public Iterable<FuzzyVariable> getInputs() {
        return inputs;
    }
    
    public Iterable<FuzzyVariable> getOutputs(){
        return outputs;
    }

    public Iterable<FuzzyRule> getRules() {
        return rules;
    }
    
    @Override
    public String toString(){
        return name;
    }

    
}
