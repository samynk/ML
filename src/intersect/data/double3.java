package intersect.data;

/**
 * Basic double3 class.
 * @author Koen
 */
public class double3 {
    
    /**
     * Creates a new double3 object with all fields set to zero.
     */
    public double3(){
        this.x = 0;
        this.y = 0;
        this.z = 0;
    }

    /**
     * Creates a new double3 object from the provided parameters.
     * @param x the x value .
     * @param y the y value.
     * @param z the z value.
     */
    public double3(double x, double y, double z){
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /**
     * The length of this double3 object = sqrt(x^2 + y^2 + z^2)
     * @return the length of this double3 object.
     */
    public double getLength(){
        return (double)Math.sqrt(x*x+y*y+z*z);
    }

    /**
     * The squared length of this double3 object = (x^2 + y^2 + z^2)
     * @return the squared length of this double3 object.
     */
    public double getLengthSquared(){
        return x*x+y*y+z*z;
    }

    /**
     * Normalizes this double3 object.
     */
    public void normalize()
    {
        double l = getLength();
        if ( l > Double.MIN_NORMAL)
        {
            x/=l;
            y/=l;
            z/=l;
        }
    }

    /**
     * A pseudo normalization of this double3 object.
     */
    public void pseudonormalize()
    {
        double l = getLengthSquared();
        if ( l > 1.0f)
        {
            x/=l;
            y/=l;
            z/=l;
        }
    }


    /**
     * Adds two double3 objects together
     * @param op1 the first double3 object to add.
     * @param op2 the second double3 object to add.
     * @return a new double3 object with the result of the add operation.
     */
    public static double3 Add(double3 op1, double3 op2){
        return new double3(op1.x+op2.x,op1.y+op2.y,op1.z+op2.z);
    }

    /**
     * Creates a linear combination of two operands. Returns the result of
     * a*op1+b*op2
     * @param a the coefficient for the first operand.
     * @param op1 the first double3 operand.
     * @param b the coefficient for the second operand.
     * @param op2 the second double3 operand.
     */
    public static double3 LinearCombination(double a, double3 op1, double b, double3 op2){
        double x = a*op1.x+b*op2.x;
        double y = a*op1.y+b*op2.y;
        double z = a*op1.z+b*op2.z;
        return new double3(x,y,z);
    }

    /**
     * Subtracts the second double3 object from the first double3 object.
     * @param op1 the first double3 object to add.
     * @param op2 the second double3 object to add.
     * @return a new double3 object with the result of the subtract operation.
     */
    public static double3 Subtract(double3 op1, double3 op2){
        return new double3(op1.x-op2.x,op1.y-op2.y,op1.z-op2.z);
    }

    /**
     * Calculates the dot product of two double3 objects.
     * @param op1 the first double3 object.
     * @param op2 the second double3 object.
     * @return the dot product of op1 and op2
     */
    public static double Dot(double3 op1, double3 op2){
        return op1.x*op2.x + op1.y*op2.y + op1.z*op2.z;
    }

    /**
     * Calculates the cross product of two double3 objects.
     * @param op1 the first double3 object.
     * @param op2 the second double3 object.
     * @return the cross product of op1 and op2.
     */
    public static double3 Cross(double3 op1, double3 op2){
        double xr = op1.y * op2.z - op1.z * op2.y;
        double yr = op1.z * op2.x - op1.x * op2.z;
        double zr = op1.x * op2.y - op1.y * op2.x;
        return new double3(xr, yr, zr);
    }

    /**
     * Creates a String representation of this object.
     * @return this double3 as a String object.
     */
    @Override
    public String toString(){
        return new String("["+x+","+y+","+z+"]");
    }

    public double x,y,z;
}