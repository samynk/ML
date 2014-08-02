/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

import java.awt.Color;
import java.util.ArrayList;
import java.util.HashMap;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 *
 * @author Koen
 */
public class FuzzyVariable {
    private HashMap<String,MemberShip> memberShipFunctions = new HashMap<>();
    private static Color[] memberShipColors;
    
    private boolean isInput;
    
    private ArrayList<ChangeListener> changeListeners = new ArrayList<>();
    
    static{
        memberShipColors = new Color[5];
        memberShipColors[0] = new Color(255,24,24,127);
        memberShipColors[1] = new Color(24,255,24,127);
        memberShipColors[2] = new Color(127,127,127,127);
        memberShipColors[3] = new Color(24,24,255,127);
        memberShipColors[4] = new Color(127,127,24,127);
        
    }
            
    
    /**
     * The name of the fuzzy variable.
     */
    private String name;
    /**
     * The input value for this fuzzy variable.
     */
    private float inputValue;
    
    /**
     * Creates a new fuzzy variable.
     * @param name the name of the fuzzy variable.
     */
    public FuzzyVariable(String name){
        this.name = name.toLowerCase();
        isInput = true;
    }
    
    /**
     * Creates a clone of the fuzzy variable.
     * @return a clone of the FuzzyVariable.
     */
    public FuzzyVariable clone(){
        FuzzyVariable clone = new FuzzyVariable(name);
        clone.isInput = isInput;
        for( String key : memberShipFunctions.keySet())
        {
            MemberShip ms = memberShipFunctions.get(key);
            clone.addMemberShip(ms.clone());
        }
        return clone;
    }
    
    public void setAsInput(){
        isInput = true;
    }
    
    public void setAsOutput(){
        isInput = false;
    }
    
    public void addChangeListener(ChangeListener listener){
        this.changeListeners.add(listener);
    }
    
    public void removeChangeListener(ChangeListener listener){
        this.changeListeners.remove(listener);
    }
    
    /**
     * Returns the name of the fuzzy variable.
     * @return 
     */
    public String getName(){
        return name;
    }
    
    /**
     * Adds a membership function to the fuzzy variable.
     * @param member  the member to add.
     */
    public void addMemberShip(MemberShip member){
        memberShipFunctions.put(member.getName(),member);
        member.setParent(this);
        member.setColor(memberShipColors[memberShipFunctions.size() % memberShipColors.length]);
    }
    
    public float getMinimum(){
        float minimum = Float.MAX_VALUE;
        for (MemberShip memberShip : this.memberShipFunctions.values())
        {
            if ( memberShip.getMinimumX() < minimum )
                minimum = memberShip.getMinimumX();
        }
        return minimum;
    }
    
    public float getMaximum(){
        float maximum = Float.MIN_VALUE;
        for (MemberShip memberShip : this.memberShipFunctions.values())
        {
            if ( memberShip.getMaximumX() > maximum )
                maximum = memberShip.getMaximumX();
        }
        return maximum;
    }
    
    /**
     * 
     * @param toEvaluate 
     */
    public float evaluateAntecedent(String member, float input, boolean not) {
        MemberShip ms = (MemberShip) memberShipFunctions.get(member);
        if (not) {
            return 1 - ms.evaluate(input);
        } else {
            return ms.evaluate(input);
        }
    }
    
    /**
     * Evaluates the member ship with the current value of this membership function. 
     * @param membership the membership function.
     * @param not inverse the result.
     * @return the membership of the current value.
     */
     public float evaluateAntecedent(String membership, boolean not) {
        MemberShip ms = (MemberShip) memberShipFunctions.get(membership);
        if ( ms == null){
            System.out.println("Membership function not found : " + membership + " for " + this.getName());
            return 0.0f;
        }
        if (not) {
            return 1 - ms.evaluate(inputValue);
        } else {
            return ms.evaluate(inputValue);
        }
    }

    /**
     * Returns the memberships of this fuzzy variable.
     * @return the membership functions of this variable.
     */
    public Iterable<MemberShip> getMemberShips() {
        return memberShipFunctions.values();
    }

    /**
     * Sets the current input value for the fuzzy variable.
     * @param value the new value for the variable;
     */
    public void setCurrentValue(float value) {
        this.inputValue = value;
        notifyInputValueChange();
    }
    
    private void notifyInputValueChange(){
        ChangeEvent ce = new ChangeEvent(this);
        for (ChangeListener cl : this.changeListeners){
            cl.stateChanged(ce);
        }
    }

    public MemberShip getMemberShip(String outputMemberShip) {
        return memberShipFunctions.get(outputMemberShip);
    }

    public float outputValueNominator;
    public float outputValueDenominator;
    
    /**
     * Adds a result with the given weights.
     * @param weight
     * @param value 
     */
    public void addOutputValue(float weight, float value) {
        outputValueNominator += weight * value;
        outputValueDenominator += weight;
    }
    
    public float getOutputValue(){
        ChangeEvent ce = new ChangeEvent(this);
        for (ChangeListener cl : this.changeListeners){
            cl.stateChanged(ce);
        }
        if ( Math.abs(outputValueDenominator) < 1e-3){
            return 0.0f;
        }else{
            float result = outputValueNominator / outputValueDenominator;
            if ( result > this.getMaximum())
                return getMaximum();
            else if ( result < this.getMinimum()){
                //System.out.println("strange : " +  outputValueNominator +"," + outputValueDenominator);
                return getMinimum();
            }
            return result;
        }
    }

    void resetOutput() {
        outputValueNominator = 0.0f;
        outputValueDenominator = 0.0f;
    }

    public boolean isInput() {
        return isInput;
    }
    
    public boolean isOutput(){
        return !isInput;
    }

    public float getInputValue() {
        return this.inputValue;
    }
    /**
     * Returns a string representation of this FuzzyVariable
     */
    @Override
    public String toString(){
        return name;
    }

    public void removeMemberShip(MemberShip ms) {
        this.memberShipFunctions.remove(ms.getName());
    }

    public void renameMemberShip(String name, String newName) {
        MemberShip ms = memberShipFunctions.get(name);
        if ( ms != null){
            memberShipFunctions.remove(name);
            memberShipFunctions.put(newName, ms);
        }
    }

    /**
     * Checks if a membership exist in this variable.
     * @param name the name to check.
     * @return true if the membership exists, false otherwise.
     */
    public boolean hasMemberShip(String name) {
        return memberShipFunctions.containsKey(name);
    }
}
