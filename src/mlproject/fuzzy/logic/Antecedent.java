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
public class Antecedent extends RulePart {

    private String variable;
    private String membership;
    private boolean negation;

    public Antecedent(String variable, String membership) {
        this.variable = variable;
        this.membership = membership;
    }

    public Antecedent() {
    }

    public String getVariable() {
        return variable;
    }

    public void setVariable(String var) {
        this.variable = var;
    }

    public String getMemberShip() {
        return membership;
    }

    public void setMemberShip(String membership) {
        this.membership = membership;
    }

    @Override
    public float evaluateAntecedent(FuzzySystem system) {
        FuzzyVariable var = system.getFuzzyInputVariable(variable);
        if (var == null) {
            System.out.println("could not find : " + variable);
        }
        float value = var.evaluateAntecedent(membership, false);
        if (negation) {
            return 1.0f - value;
        } else {
            return value;
        }
    }

    @Override
    public float evaluateAntecedentVerbally(FuzzySystem system) {
        FuzzyVariable var = system.getFuzzyInputVariable(variable);
        if (var == null) {
            System.out.println("could not find : " + variable);
        }
        float result = var.evaluateAntecedent(membership, false);
        if (negation) {
            result = 1.0f - result;
        }

        System.out.println("Antecedent : " + var.getName() + "," + membership + "=" + result);
        return result;
    }

    @Override
    public float evaluateTestValue(FuzzySystem system) {
        FuzzyVariable var = system.getFuzzyInputVariable(variable);
        float value=  var.getMemberShip(membership).getInputTestValue();
        if (negation) {
            return 1.0f - value;
        } else {
            return value;
        }
    }

    @Override
    public void getInputs(FuzzySystem system, ArrayList<MemberShip> inputs) {
        FuzzyVariable var = system.getFuzzyInputVariable(variable);
        inputs.add(var.getMemberShip(membership));
    }

    public void setNegation(boolean negation) {
        this.negation = negation;
    }
}
