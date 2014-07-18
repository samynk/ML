package intersect.data;

/**
 * @author Koen Samyn
 */
public class double2 {

    public double2(){
    }

    public double2(double x, double y){
        this.x = x;
        this.y = y;
    }

    public double getX(){
        return x;
    }

    public double getY(){
        return y;
    }

    /**
     * The length of this double2 object = sqrt(x^2 + y^2 + z^2)
     * @return the length of this double2 object.
     */
    public double getLength(){
        return (double)Math.sqrt(x*x+y*y);
    }

    /**
     * The squared length of this double2 object = (x^2 + y^2 + z^2)
     * @return the squared length of this double2 object.
     */
    public double getLengthSquared(){
        return x*x+y*y;
    }

    /**
     * Adds two double3 objects together
     * @param op1 the first double2 object to add.
     * @param op2 the second double2 object to add.
     * @return a new double2 object with the result of the add operation.
     */
    public static double2 Add(double2 op1, double2 op2){
        return new double2(op1.x+op2.x,op1.y+op2.y);
    }

    /**
     * Creates a linear combination of two operands. Returns the result of
     * a*op1+b*op2
     * @param a the coefficient for the first operand.
     * @param op1 the first double2 operand.
     * @param b the coefficient for the second operand.
     * @param op2 the second double2 operand.
     */
    public static double2 LinearCombination(double a, double2 op1, double b, double2 op2){
        double x = a*op1.x+b*op2.x;
        double y = a*op1.y+b*op2.y;
        return new double2(x,y);
    }

    /**
     * Subtracts the second double2 object from the first double3 object.
     * @param op1 the first double2 object to add.
     * @param op2 the second double2 object to add.
     * @return a new double3 object with the result of the subtract operation.
     */
    public static double2 Subtract(double2 op1, double2 op2){
        return new double2(op1.x-op2.x,op1.y-op2.y);
    }

    /**
     * Calculates the dot product of two double3 objects.
     * @param op1 the first double2 object.
     * @param op2 the second double2 object.
     * @return the dot product of op1 and op2
     */
    public static double Dot(double2 op1, double2 op2){
        return op1.x*op2.x + op1.y*op2.y;
    }

    /**
     * Calculates the cross product of two double2 objects. Because the coordinates
     * are xy coordinates, the cross product will produces a double3 object.
     * @param op1 the first double2 object.
     * @param op2 the second double2 object.
     * @return the cross product of op1 and op2.
     */
    public static double3 Cross(double2 op1, double2 op2){
        double zr = op1.x * op2.y - op1.y * op2.x;
        return new double3(0,0, zr);
    }

    /**
     * Creates a String representation of this object.
     * @return this double3 as a String object.
     */
    @Override
    public String toString(){
        return new String("["+x+","+y+"]");
    }


    public double x, y;
}
