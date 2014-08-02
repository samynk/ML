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
    
    /**
     * Creates a clone of this membership function.
     * @return the clone of this membership function.
     */
    @Override
    public MemberShip clone(){
        return new LeftSigmoidMemberShip(center,right,getName());
    }
    
    private void recalc(){
        float factor = (float)Math.log(1/.99 - 1);
        
        rightB = center + (right-center)/2;
        rightA = factor / (rightB-center);
    }
    
    /**
     * Sets the center of this left sigmoid membership.
     * @param center the new center for this left sigmoid membership.
     */
    public void setCenter(float center) {
        this.center = center;
        recalc();
        notifyListeners();
    }
    
    /**
     * Gets the center of this left sigmoid membership.
     * @return the center of this membership function.
     */
    public float getCenter(){
        return center;
    }
    
    /**
     * Sets the right value of this left sigmoid membership.
     * @param right the new right value for this left sigmoid membership.
     */
    public void setRight(float right){
        this.right = right;
        recalc();
        notifyListeners();
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
    
   
    
    public float getA(){
        return rightA;
    }
    public float getB(){
        return rightB;
    }
    
    @Override
     public String toString(){
        return "Shoulder left(" + center + " " + right +")";
    }
}
