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
    private int rulegroup;

    public FuzzyRuleBlock(FuzzySystem parent, int rulegroup) {
        this.rulegroup = rulegroup;
        this.parent = parent;
    }

    public void addFuzzyRule(FuzzyRule rule) {
        this.rules.add(rule);
        for (Antecedent a : rule.getAntecedents()) {
            String var = a.getVariable();
            FuzzyVariable input = parent.getFuzzyInputVariable(var);
            if (!inputs.contains(input)) {
                inputs.add(input);
            }
        }
        String ovar = rule.getOutputVariable();
        FuzzyVariable output = parent.getFuzzyOutputVariable(ovar);
        if (!outputs.contains(output)) {
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
}
