/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

/**
 * This class models a membership function with the
 * help of logistic sigmoid functions.
 * Because the sigmoid function has 2 assympotes the values
 * the left center and right properties of the membership are
 * calculated by setting :
 * 1/(1+Exp[-a(x+b)] == 1 for the maximum membership.
 * and 1/(1+Exp]-a(x+b]== 0 for the minimum membership.
 * 
 * The membership function can be asymetric.
 * 
 * 
 * 
 * @author Koen
 */
public class SigmoidMemberShip extends MemberShip implements Cloneable{
    private float center; // equals b in the equations
    private float left;
    private float right;
    
    private float leftA;
    private float leftB;
    
    private float rightA;
    private float rightB;
    
    /**
     * Creates a new SigmoidMemberShip by subtracting two sigmoid functions.
     * @param left the left position of the sigmoid member ship.
     * @param center the center position of the sigmoid member ship.
     * @param right the right position of the sigmoid member ship.
     * @param name the name of the membership.
     */
    public SigmoidMemberShip(float left, float center, float right, String name){
        super(name);
        this.center = center;
        this.left = left;
        this.right = right;
        
        float factor = (float)Math.log(1/.99 - 1);
        
        leftB = left + (center-left)/2;
        //leftA = factor/ (center-leftB);
        leftA = factor / (leftB-left);
        
        rightB = center + (right-center)/2;
        rightA = factor / (rightB-center);
    }
    
    private void recalc(){
        float factor = (float)Math.log(1/.99 - 1);
        
        leftB = left + (center-left)/2;
        leftA = factor/ (leftB-left);
        
        rightB = center + (right-center)/2;
        rightA = factor / (rightB-center);
    }
    
    @Override
    public float evaluate(float x){
       float leftValue = 1/(1+(float)Math.exp(leftA*(x-leftB)));
       float rightValue = 1/(1+(float)Math.exp(rightA*(x-rightB)));
       return  leftValue - rightValue;
    }
    
    @Override
      public float getMinimumX(){
        return left;
    }
    
    @Override
    public float getMaximumX(){
        return right;
    }
    
    @Override
    public void move(float dx) {
        this.center += dx;
        this.left += dx;
        this.right += dx;
        recalc();
    }
    
    public float getLeft(){
        return left;
    }
    
    public float getCenter(){
        return center;
    }
    
    public float getRight(){
        return right;
    }
    
    public String toString(){
        return "Triangular(" + left + " " + center + " " + right+")";
    }
    
    public float getLeftA() {
        return leftA;
    }

    public float getLeftB() {
        return leftB;
    }
    
    public float getRightA() {
        return rightA;
    }

    public float getRightB() {
        return rightB;
    }
    
    @Override
    public Object clone() throws CloneNotSupportedException {
        return new SigmoidMemberShip(this.left,this.center,this.right,getName());
    }
}
