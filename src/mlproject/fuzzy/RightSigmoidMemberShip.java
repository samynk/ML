/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

/**
 * This class describes a right sigmoid membership function.
 * @author Koen Samyn
 */
public class RightSigmoidMemberShip extends MemberShip implements Cloneable {

    private float center;
    private float left;
    private float leftA;
    private float leftB;

    /**
     * Creates a new SigmoidMemberShip by subtracting two sigmoid functions.
     * @param left the left position of the sigmoid member ship.
     * @param center the center position of the sigmoid member ship.
     * @param right the right position of the sigmoid member ship.
     * @param name the name of the membership.
     */
    public RightSigmoidMemberShip(float left, float center, String name) {
        super(name);
        this.center = center;
        this.left = left;

        recalc();
    }

    private void recalc() {
        float factor = (float) Math.log(1 / .99 - 1);

        leftB = left + (center - left) / 2;
        leftA = factor / (center - leftB);
    }

    /**
     * Gets the center of this left sigmoid membership.
     * @return the center of this membership function.
     */
    public float getCenter() {
        return center;
    }

    /**
     * Gets the left of this right sigmoid membership.
     * @return the left side of this membership function.
     */
    public float getLeft() {
        return left;
    }

    @Override
    public float evaluate(float x) {
        float leftValue = 1 / (1 + (float) Math.exp(leftA * (x - leftB)));
        return leftValue;
    }

    public float getMinimumX() {
        return left;
    }

    public float getMaximumX() {
        return center + (center - left);
    }

    public void move(float dx) {
        this.center += dx;
        this.left += dx;
        recalc();
    }

    public float getA() {
        return leftA;
    }

    public float getB() {
        return leftB;
    }

    public String toString() {
        return "Shoulder right(" + left + " " + center + ")";
    }
    
       @Override
    public Object clone() throws CloneNotSupportedException {
        return new RightSigmoidMemberShip(this.left,this.center,getName());
    }
}
