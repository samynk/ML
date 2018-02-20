/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

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
    public void testConvolution() {
        fmatrix input = new fmatrix(2048, 2048);
        input.applyCellFunction(c -> c.row * 7 + c.column);
        fmatrix filter = new fmatrix(5, 5);
        filter.applyCellFunction(c -> c.row == 0 ? .1f : .2f);
        fmatrix output1 = new fmatrix(2044, 2044);

        long start1 = System.currentTimeMillis();
        FMaxtrixOpGpu.convolve(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("GPU time : " + (end1 - start1));

        fmatrix output2 = new fmatrix(2044, 2044);
        long start = System.currentTimeMillis();
        FMatrixOpCpu cpu = new FMatrixOpCpu();
        cpu.convolve(input, filter, 1, output2);
        long end = System.currentTimeMillis();

        System.out.println("CPU time : " + (end - start));

        for (int r = 0; r < output1.getNrOfRows(); ++r) {
            for (int c = 0; c < output1.getNrOfColumns(); ++c) {
                float v1 = output1.get(r, c);
                float v2 = output2.get(r, c);
                assertEquals("error on :" + r + "," + c, v1, v2, 0.01f);
            }
        }
    }
}
