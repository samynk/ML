/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

import java.awt.Color;

/**
 *
 * @author Koen
 */
public class MemberShip implements Cloneable {

    private String name;
    private FuzzyVariable parent;
    private float inputTestValue;

    public MemberShip(String name) {
        this.name = name.toLowerCase();
    }

    public float evaluate(float x) {
        return 1.0f;
    }

    public String getName() {
        return name;
    }

    public float getMinimumX() {
        return -1.0f;
    }

    public float getMaximumX() {
        return 1.0f;
    }
    Color color;

    public void setColor(Color color) {
        this.color = color;
    }

    public Color getColor() {
        return color;
    }

    public void move(float dx) {
    }

    public String toString() {
        return "MemberShip base class";
    }

    /**
     * @return the parent
     */
    public FuzzyVariable getParent() {
        return parent;
    }

    /**
     * @param parent the parent to set
     */
    public void setParent(FuzzyVariable parent) {
        this.parent = parent;
    }

    public float getInputTestValue() {
        return inputTestValue;
    }

    public void setInputTestValue(float inputTestValue) {
        this.inputTestValue = inputTestValue;
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        return new MemberShip(this.name);
    }
}
