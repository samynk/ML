package dae.neuralnet.opencl;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.tmatrix;
import org.jblas.FloatMatrix;
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
public class TestJBlas {

    public TestJBlas() {
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

    // TODO add test methods here.
    // The methods must be annotated with annotation @Test. For example:
    //
    @Test
    public void testInit() {
        fmatrix a = fmatrix.random(4, 3, -2, 2);
        imatrix ta = new tmatrix(a);
        fmatrix b = fmatrix.random(4, 2, -2, 2);
        fmatrix c = fmatrix.ones(3, 2);

        float alpha = .5f;
        float beta = .2f;

        imatrix ac = a.copy();
        imatrix tac = new tmatrix(ac);
        imatrix bc = b.copy();
        imatrix cc = c.copy();

        System.out.println("Result:");
        fmatrix.sgemm(alpha, ta, b, beta, c);
        System.out.println(c.toString());
        System.out.println("_______\n\n\n");

        imatrix ctemp = fmatrix.multiply(tac, bc);
        ctemp.multiply(alpha);

        cc.multiply(beta);
        fmatrix.dotadd(cc, ctemp, cc);

        System.out.println("Result:");
        System.out.println(cc);
    }
}
