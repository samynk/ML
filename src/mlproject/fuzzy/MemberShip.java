/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

import java.awt.Color;
import java.util.ArrayList;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 *
 * @author Koen
 */
public class MemberShip {

    private String name;
    private FuzzyVariable parent;
    private float inputTestValue;
    private ArrayList<ChangeListener> listeners;
    private ChangeEvent changeEvent;

    public MemberShip(String name) {
        this.name = name.toLowerCase();
        changeEvent = new ChangeEvent(this);
    }
    
    @Override
    public MemberShip clone(){
        return new MemberShip(name);
    }

    public void addChangeListener(ChangeListener cl) {
        if (listeners == null) {
            listeners = new ArrayList<>();

        }
        listeners.add(cl);
    }

    public void removeChangeListener(ChangeListener cl) {
        if (listeners == null) {
            return;
        }
        listeners.remove(cl);
    }

    protected void notifyListeners() {
        if (listeners != null) {
            for (ChangeListener cl : listeners) {
                cl.stateChanged(changeEvent);
            }
        }
    }

    public float evaluate(float x) {
        return 1.0f;
    }

    public String getName() {
        return name;
    }

    /**
     * Sets the name of this membership.
     */
    public void setName(String newName) {
        if (checkName(newName)) {
            if (parent != null) {
                parent.renameMemberShip(name, newName);
            }
            this.name = newName;
            notifyListeners();
        }
    }

    public boolean checkName(String name) {
        if (name.length() == 0) {
            return false;
        } else if (parent != null) {
            return !parent.hasMemberShip(name);
        } else {
            return true;
        }
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
}
