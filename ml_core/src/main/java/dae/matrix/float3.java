package dae.matrix;

/**
 * 
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class float3 {
    
    /**
     * Creates a new float3 object with all fields set to zero.
     */
    public float3(){
        this.x = 0;
        this.y = 0;
        this.z = 0;
    }

    /**
     * Creates a new float3 object from the provided parameters.
     * @param x the x value .
     * @param y the y value.
     * @param z the z value.
     */
    public float3(float x, float y, float z){
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /**
     * The length of this float3 object = sqrt(x^2 + y^2 + z^2)
     * @return the length of this float3 object.
     */
    public float getLength(){
        return (float)Math.sqrt(x*x+y*y+z*z);
    }

    /**
     * The squared length of this float3 object = (x^2 + y^2 + z^2)
     * @return the squared length of this float3 object.
     */
    public float getLengthSquared(){
        return x*x+y*y+z*z;
    }

    /**
     * Normalizes this float3 object.
     */
    public void normalize()
    {
        float l = getLength();
        if ( l > Float.MIN_NORMAL)
        {
            x/=l;
            y/=l;
            z/=l;
        }
    }

    /**
     * A pseudo normalization of this float3 object.
     */
    public void pseudonormalize()
    {
        float l = getLengthSquared();
        if ( l > 1.0f)
        {
            x/=l;
            y/=l;
            z/=l;
        }
    }


    /**
     * Adds two float3 objects together
     * @param op1 the first float3 object to add.
     * @param op2 the second float3 object to add.
     * @return a new float3 object with the result of the add operation.
     */
    public static float3 Add(float3 op1, float3 op2){
        return new float3(op1.x+op2.x,op1.y+op2.y,op1.z+op2.z);
    }

    /**
     * Creates a linear combination of two operands. Returns the result of
     * a*op1+b*op2
     * @param a the coefficient for the first operand.
     * @param op1 the first float3 operand.
     * @param b the coefficient for the second operand.
     * @param op2 the second float3 operand.
     */
    public static float3 LinearCombination(float a, float3 op1, float b, float3 op2){
        float x = a*op1.x+b*op2.x;
        float y = a*op1.y+b*op2.y;
        float z = a*op1.z+b*op2.z;
        return new float3(x,y,z);
    }

    /**
     * Subtracts the second float3 object from the first float3 object.
     * @param op1 the first float3 object to add.
     * @param op2 the second float3 object to add.
     * @return a new float3 object with the result of the subtract operation.
     */
    public static float3 Subtract(float3 op1, float3 op2){
        return new float3(op1.x-op2.x,op1.y-op2.y,op1.z-op2.z);
    }

    /**
     * Calculates the dot product of two float3 objects.
     * @param op1 the first float3 object.
     * @param op2 the second float3 object.
     * @return the dot product of op1 and op2
     */
    public static float Dot(float3 op1, float3 op2){
        return op1.x*op2.x + op1.y*op2.y + op1.z*op2.z;
    }

    /**
     * Calculates the cross product of two float3 objects.
     * @param op1 the first float3 object.
     * @param op2 the second float3 object.
     * @return the cross product of op1 and op2.
     */
    public static float3 Cross(float3 op1, float3 op2){
        float xr = op1.y * op2.z - op1.z * op2.y;
        float yr = op1.z * op2.x - op1.x * op2.z;
        float zr = op1.x * op2.y - op1.y * op2.x;
        return new float3(xr, yr, zr);
    }

    /**
     * Creates a String representation of this object.
     * @return this float3 as a String object.
     */
    @Override
    public String toString(){
        return "["+x+","+y+","+z+"]";
    }

    public float x,y,z;
}