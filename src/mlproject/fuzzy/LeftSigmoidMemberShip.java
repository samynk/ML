/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

/**
 *
 * @author Koen
 */
public class LeftSigmoidMemberShip extends MemberShip implements Cloneable{
    private float center; 
    private float right;
    
    private float rightA;
    private float rightB;
    
    
    
    /**
     * Creates a new SigmoidMemberShip by subtracting two sigmoid functions.
     * @param left the left position of the sigmoid member ship.
     * @param center the center position of the sigmoid member ship.
     * @param right the right position of the sigmoid member ship.
     * @param name the name of the membership.
     */
    public LeftSigmoidMemberShip(float center, float right, String name){
        super(name);
        this.center = center;
        this.right = right;
        
        recalc();
    }
    
    private void recalc(){
        float factor = (float)Math.log(1/.99 - 1);
        
        rightB = center + (right-center)/2;
        rightA = factor / (rightB-center);
    }
    
    /**
     * Gets the center of this left sigmoid membership.
     * @return the center of this membership function.
     */
    public float getCenter(){
        return center;
    }
    
    /**
     * Gets the right of this left sigmoid membership.
     * @return the right side of this membership function.
     */
    public float getRight(){
        return right;
    }
    
    @Override
    public float evaluate(float x){
       
       float rightValue = 1/(1+(float)Math.exp(rightA*(x-rightB)));
       return  1 - rightValue;
    }
    
    public float getMinimumX(){
        return center - (right - center);
    }
    
    public float getMaximumX(){
        return right;
    }
    
    public void move(float dx) {
        this.center += dx;
        this.right += dx;
        recalc();
    }
    
    public String toString(){
        return "Shoulder left(" + center + " " + right +")";
    }
    
    public float getA(){
        return rightA;
    }
    public float getB(){
        return rightB;
    }
    
      @Override
    public Object clone() throws CloneNotSupportedException {
        return new LeftSigmoidMemberShip(this.center,this.right,getName());
    }
}
