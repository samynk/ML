/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

/**
 *
 * @author Koen
 */
public class TrapezoidMemberShip extends MemberShip implements Cloneable {

    private float centerleft;
    private float centerright;
    private float left;
    private float right;
    private float leftA;
    private float leftB;
    private float rightA;
    private float rightB;

    /**
     * Creates a new TrapezoidMemberShip function
     *
     * @param left the left value of the membership function.
     * @param centerleft the centerleft value of the membership function.
     * @param centerright the centerright value of the membership function.
     * @param right the right value of the membership function.
     */
    public TrapezoidMemberShip(float left, float centerleft, float centerright, float right, String name) {
        super(name);
        this.left = left;
        this.centerleft = centerleft;
        this.centerright = centerright;
        this.right = right;

        float factor = (float) Math.log(1 / .99 - 1);

        leftB = left + (centerleft - left) / 2;
        //leftA = factor/ (center-leftB);
        leftA = factor / (leftB - left);

        rightB = centerright + (right - centerright) / 2;
        rightA = factor / (rightB - centerright);
    }
    
    @Override
    public MemberShip clone() {
        return new TrapezoidMemberShip(this.left, this.centerleft, this.centerright, this.right, getName());
    }

    private void recalc() {
        float factor = (float) Math.log(1 / .99 - 1);

        leftB = left + (centerleft - left) / 2;
        leftA = factor / (leftB - left);

        rightB = centerright + (right - centerright) / 2;
        rightA = factor / (rightB - centerright);
    }

    @Override
    public float evaluate(float x) {
        float leftValue = 1 / (1 + (float) Math.exp(leftA * (x - leftB)));
        float rightValue = 1 / (1 + (float) Math.exp(rightA * (x - rightB)));
        return leftValue - rightValue;
    }

    @Override
    public float getMinimumX() {
        return left;
    }

    @Override
    public float getMaximumX() {
        return right;
    }

    @Override
    public void move(float dx) {
        this.centerleft += dx;
        this.centerright += dx;
        this.left += dx;
        this.right += dx;
        recalc();
    }

    public void setLeft(float left) {
        this.left = left;
        recalc();
        notifyListeners();
    }

    public float getLeft() {
        return left;
    }

    public void setCenterLeft(float centerleft) {
        this.centerleft = centerleft;
        recalc();
        notifyListeners();
    }

    public float getCenterLeft() {
        return centerleft;
    }

    public void setCenterRight(float centerright) {
        this.centerright = centerright;
        recalc();
        notifyListeners();
    }

    public float getCenterRight() {
        return centerright;
    }
    
    public void setRight(float right){
        this.right = right;
        recalc();
        notifyListeners();
    }

    public float getRight() {
        return right;
    }

    public String toString() {
        return "Trapezium(" + left + " " + centerleft + " " + centerright + " " + right + ")";
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

    
}
