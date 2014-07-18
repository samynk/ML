/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.animation;

/**
 * Defines a controller for a bone.
 * @author samyn_000
 */
public interface Controller {
    public void evaluate(float time);
    public void setInput(String name, float value);
    public float getOutput(String name);
    
    public boolean isAbsoluteAngle();
}
