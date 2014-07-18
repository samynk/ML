package intersect.data;

import java.util.Random;

/**
 * @author Koen Samyn
 */
public class double2x2 {

    public double a1, a2, b1, b2;

    public double2x2(){
    
    }

    public double2x2(double a1, double a2, double b1, double b2) {
        setElements(a1, a2, b1, b2);
    }

    public double2x2 inverse() {
        double d = a1 * b2 - a2 * b1;
        return new double2x2(b2 / d, -a2 / d, -b1 / d, a1 / d);
    }

    public void inverse(double2x2 result){
        double d = a1 * b2 - a2 * b1;
        result.setElements(b2 / d, -a2 / d, -b1 / d, a1 / d);
    }

    public double2 multiply(double2 op) {
        return new double2(a1 * op.x + a2 * op.y, b1 * op.x + b2 * op.y);
    }

    public void multiply(double2 result, double2 op){
        double rx = a1 * op.x + a2 * op.y;
        double ry = b1 * op.x + b2 * op.y;
        result.x = rx;
        result.y = ry;
    }

    public double2 multiply(double r1, double r2) {
        return new double2(a1 * r1 + a2 * r2, b1 * r1 + b2 * r2);
    }

    public void multiply(double2 result, double r1, double r2){
        double rx = a1 * r1 + a2 * r2;
        double ry = b1 * r1 + b2 * r2;
        result.x = rx;
        result.y = ry;
    }

    /**
     * solve the equation :
     * a1*x + a2*y = c1
     * b1*x + b2*y = c2
     * @param x the x parameter to solve
     * @param y the y paramter to solve
     */
    public double2 solve(double c1, double c2)
    {
        double2 result = new double2();
        if ( solve(result, c1, c2 ))
            return result;
        else
            return null;
    }

    /**
     * solve the equation :
     * a1*x + a2*y = c1
     * b1*x + b2*y = c2
     * @param x the x parameter to solve
     * @param y the y paramter to solve
     */
    public boolean solve(double2 result, double c1,double c2)
    {
        double D = a1*b2 - a2*b1;
        if ( Math.abs(D) > Double.MIN_NORMAL )
        {
            if ( Math.abs(a1) > Double.MIN_NORMAL )
            {
                result.y = (a1*c2 - b1*c1)/D;
                result.x = (c1 - a2*result.y)/a1;
            }else{
                result.y = c1/a2;
                result.x = (c2 - b2*result.y)/b1;
            }
            return true;
        }
        return false;
    }

    public void setRow1(double a1, double a2) {
        this.a1 = a1;
        this.a2 = a2;
    }

    public void setRow2(double b1, double b2) {
        this.b1 = b1;
        this.b2 = b2;
    }

    public void setColumn1(double a1, double b1) {
        this.a1 = a1;
        this.b1 = b1;
    }

    public void setColumn2(double a2, double b2) {
        this.a2 = a2;
        this.b2 = b2;
    }

    public void setElements(double a1, double a2, double b1, double b2) {
        this.a1 = a1;
        this.a2 = a2;
        this.b1 = b1;
        this.b2 = b2;
    }

    @Override
    public String toString()
    {
        return "["+a1+","+a2+";"+b1+","+b2+"]";
    }

    public static void main(String[] args) {
        double result2 = 1 / Double.MIN_NORMAL;
      
        double2x2 m = new double2x2(-4755.489,843.5676,-756.9105,134.03232);
        double2x2 inverse = m.inverse();
        double2 solution2 = inverse.multiply(-2953.724,1093.669);

        double2 check2 = new double2();
        m.multiply(check2,solution2);

        System.out.println(solution2);
        System.out.println(check2);

        System.out.println("Errors : [" + (check2.x +2953.724)+","+(check2.y-1093.669)+"]");
    }

    
}
