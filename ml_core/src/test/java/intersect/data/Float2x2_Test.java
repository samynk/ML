/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package intersect.data;

import dae.matrix.float2;
import dae.matrix.float2x2;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class Float2x2_Test {

    public Float2x2_Test() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    @Test
    public void solveTest1() {
        float2x2 m = new float2x2(5, 4, 13, 7);
        float2 result = m.solve(12, 18);

        assertEquals(result.x, -0.70588225, 0.0001f);
        assertEquals(result.y, 3.8823528, 0.0001f);

        float2 check = new float2();
        m.multiply(check, result);

        float2x2 inverse = m.inverse();
        float2 solution2 = inverse.multiply(12, 18);

        float2 check2 = new float2();
        m.multiply(check2, solution2);

        assertEquals(check.x, check2.x, 0.00001f);
        assertEquals(check.y, check2.y, 0.00001f);
    }

    @Test
    public void solveTest2() {
        float2x2 temp = new float2x2();
        float2x2 itemp = new float2x2();

        float2 solution = new float2();
        float2 check = new float2();
        Random r = new Random(System.currentTimeMillis());
        long start = System.currentTimeMillis();
        // test with inverse
        int matrixErrors = 0;
        int solverErrors = 0;
        for (int i = 1; i < 100; ++i) {
            float a1 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float a2 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float b1 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float b2 = r.nextFloat() + (r.nextInt(10000) - 5000);

            temp.setElements(a1, a2, b1, b2);
            temp.inverse(itemp);

            float c1 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float c2 = r.nextFloat() + (r.nextInt(10000) - 5000);

            itemp.multiply(solution, c1, c2);

            temp.multiply(check, solution);

            float error1 = Math.abs(check.x - c1);
            float error2 = Math.abs(check.y - c2);

            assertEquals(check.x, c1, 0.1f);
            assertEquals(check.y, c2, 0.1f);

            if (error1 > .1f || error2 > .1f) {
                System.out.println("error with :" + temp);
                System.out.println("inverse is :" + itemp);
                System.out.println("and C : [" + c1 + ";" + c2 + "]");
                System.out.println("Solution is : " + solution);
                System.out.println("Error1 is :" + error1);
                System.out.println("Error2 is :" + error2);
                matrixErrors++;
            }
        }
        long end = System.currentTimeMillis();
        System.out.println("inverse method : " + (end - start));

        System.out.println("#Beginning solve test");
        System.out.println("#####################");
        System.out.println("#####################");
        System.out.println("#####################");
        System.out.println("#####################");
        long start2 = System.currentTimeMillis();

        // test with inverse
        for (int i = 1; i < 100; ++i) {
            float a1 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float a2 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float b1 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float b2 = r.nextFloat() + (r.nextInt(10000) - 5000);

            temp.setElements(a1, a2, b1, b2);

            float c1 = r.nextFloat() + (r.nextInt(10000) - 5000);
            float c2 = r.nextFloat() + (r.nextInt(10000) - 5000);

            temp.solve(solution, c1, c2);

            temp.multiply(check, solution);

            float error1 = Math.abs(check.x - c1);
            float error2 = Math.abs(check.y - c2);

            assertEquals(check.x, c1, 0.1f);
            assertEquals(check.y, c2, 0.1f);

            if (error1 > .1f || error2 > .1f) {
                System.out.println("error with :" + temp);

                System.out.println("and c1 : " + c1);
                System.out.println("and c2 : " + c2);

                System.out.println("Solution is : " + solution);

                System.out.println("Error1 is :" + error1);
                System.out.println("Error2 is :" + error2);

                solverErrors++;
            }
        }
        long end2 = System.currentTimeMillis();
        System.out.println("Solve method : " + (end2 - start2));

        System.out.println("Matrix errors : " + matrixErrors);
        System.out.println("Solver errors : " + solverErrors);

    }
}
