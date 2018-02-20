/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrx.cpu;

import dae.matrix.cpu.FMatrixOpCpu;
import dae.matrix.fmatrix;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FMatrixOpTest {
    
    public FMatrixOpTest() {
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
     public void testConvolve() {
         fmatrix input = new fmatrix(7,7);
         
         fmatrix filter = new fmatrix(5,5);
         fmatrix output = new fmatrix(3,3);
         
         FMatrixOpCpu cpu = new FMatrixOpCpu();
         cpu.convolve(input, filter, 1, output);
         
         System.out.println(output);
     }
}
