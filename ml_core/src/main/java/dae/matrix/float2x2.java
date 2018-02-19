package dae.matrix;

import java.util.Random;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class float2x2 {

    public float a1, a2, b1, b2;

    public float2x2() {

    }

    public float2x2(float a1, float a2, float b1, float b2) {
        setElements(a1, a2, b1, b2);
    }

    public float2x2 inverse() {
        float d = a1 * b2 - a2 * b1;
        return new float2x2(b2 / d, -a2 / d, -b1 / d, a1 / d);
    }

    public void inverse(float2x2 result) {
        float d = a1 * b2 - a2 * b1;
        result.setElements(b2 / d, -a2 / d, -b1 / d, a1 / d);
    }

    public float2 multiply(float2 op) {
        return new float2(a1 * op.x + a2 * op.y, b1 * op.x + b2 * op.y);
    }

    public void multiply(float2 result, float2 op) {
        float rx = a1 * op.x + a2 * op.y;
        float ry = b1 * op.x + b2 * op.y;
        result.x = rx;
        result.y = ry;
    }

    public float2 multiply(float r1, float r2) {
        return new float2(a1 * r1 + a2 * r2, b1 * r1 + b2 * r2);
    }

    public void multiply(float2 result, float r1, float r2) {
        float rx = a1 * r1 + a2 * r2;
        float ry = b1 * r1 + b2 * r2;
        result.x = rx;
        result.y = ry;
    }

    /**
     * solve the equation : a1*x + a2*y = c1 b1*x + b2*y = c2
     *
     * @param c1 the c1 parameter
     * @param c2 the c2 parameter
     * @return the x and y value of the result in a float2 object.
     */
    public float2 solve(float c1, float c2) {
        float2 result = new float2();
        if (solve(result, c1, c2)) {
            return result;
        } else {
            return null;
        }
    }

    /**
     * solve the equation : a1*x + a2*y = c1 b1*x + b2*y = c2
     *
     * @param c1 the c1 parameter
     * @param c2 the c2 parameter
     * @param result stores the x and y value of the result.
     * @return true if a result was found, false otherwise.
     */
    public boolean solve(float2 result, float c1, float c2) {
        float D = a1 * b2 - a2 * b1;
        if (Math.abs(D) > Float.MIN_NORMAL) {
            if (Math.abs(a1) > Float.MIN_NORMAL) {
                result.y = (a1 * c2 - b1 * c1) / D;
                result.x = (c1 - a2 * result.y) / a1;
            } else {
                result.y = c1 / a2;
                result.x = (c2 - b2 * result.y) / b1;
            }
            return true;
        }
        return false;
    }

    /**
     * Sets the first row in the matrix to a1 and a2.
     *
     * @param a1 the new value for the first cell in the first row.
     * @param a2 the new value for the second cell in the first row.
     */
    public void setRow1(float a1, float a2) {
        this.a1 = a1;
        this.a2 = a2;
    }

    /**
     * Sets the second row in the matrix to a1 and a2.
     *
     * @param b1 the new value for the first cell in the second row.
     * @param b2 the new value for the second cell in the second row.
     */
    public void setRow2(float b1, float b2) {
        this.b1 = b1;
        this.b2 = b2;
    }

    /**
     * Sets the first column in the matrix to a1 and b1.
     *
     * @param a1 the new value for the first cell in the first row.
     * @param b1 the new value for the first cell in the second row.
     */
    public void setColumn1(float a1, float b1) {
        this.a1 = a1;
        this.b1 = b1;
    }

    /**
     * Sets the second column in the matrix to a2 and b2.
     *
     * @param a2 the new value for the second cell in the first row.
     * @param b2 the new value for the second cell in the second row.
     */
    public void setColumn2(float a2, float b2) {
        this.a2 = a2;
        this.b2 = b2;
    }

    /**
     * Sets all the elements in this 2x2 matrix.
     * @param a1 the new value for the element at row:1, column:1.
     * @param a2 the new value for the element at row:1, column:2.
     * @param b1 the new value for the element at row:2, column:1.
     * @param b2 the new value for the element at row:2, column:2.
     */
    public final void setElements(float a1, float a2, float b1, float b2) {
        this.a1 = a1;
        this.a2 = a2;
        this.b1 = b1;
        this.b2 = b2;
    }

    /**
     * A string representation of this matrix.
     * @return this matrix as a string.
     */
    @Override
    public String toString() {
        return "[" + a1 + "," + a2 + ";" + b1 + "," + b2 + "]";
    }
}
