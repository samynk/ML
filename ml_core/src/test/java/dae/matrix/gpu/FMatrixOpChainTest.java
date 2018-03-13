/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.cpu.FMatrixOpCpu;
import dae.matrix.fmatrix;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FMatrixOpChainTest {

    private FMatrixOpGpu gpu;
    private FMatrixOpCpu cpu;

    public FMatrixOpChainTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
        cpu = new FMatrixOpCpu();
        gpu = new FMatrixOpGpu();
    }

    @After
    public void tearDown() {
    }

    @Test
    public void testDotChaining() {
        Random r = new Random(System.currentTimeMillis());
        int rows = r.nextInt(20) + 5;
        int columns = r.nextInt(30) + 10;
        fmatrix s1 = new fmatrix(rows, columns);
        fmatrix.randomize(s1, r, -1, +1);

        fmatrix s2 = new fmatrix(rows, columns);
        fmatrix.randomize(s2, r, -2, +2);

        fmatrix s3 = new fmatrix(rows, columns);
        fmatrix.randomize(s3, r, -2, +2);

        fmatrix cpuresult1 = new fmatrix(rows, columns);
        fmatrix cpuresult2 = new fmatrix(rows, columns);

        cpu.dotadd(cpuresult1, s1, s2);
        cpu.dotmultiply(cpuresult2, cpuresult1, s3);

        fmatrix gpuresult1 = new fmatrix(rows, columns);
        fmatrix gpuresult2 = new fmatrix(rows, columns);

        gpu.dotadd(gpuresult1, s1, s2);
        gpu.dotmultiply(gpuresult2, gpuresult1, s3);

        gpuresult2.sync();
        MatrixTestUtil.assertMatrixEquals(cpuresult2, gpuresult2);

    }

    @Test
    public void testRectCopy() {
        // test of copying source to a destination with
        // dimensions.
        Random r = new Random(System.currentTimeMillis());

        int n = 5 * 5;
        int batchSize = 10;

        fmatrix src = new fmatrix(1, n, 1, batchSize);
        fmatrix.randomize(src, r, -5, 5);
        fmatrix cpu_dst = new fmatrix(1, n + 1, 1, batchSize);
        fmatrix gpu_dst = new fmatrix(1, n + 1, 1, batchSize);
        
        cpu.copyInto(src, cpu_dst);
        gpu.copyInto(src, gpu_dst);
        
        gpu_dst.sync();
        
        MatrixTestUtil.assertMatrixEquals(gpu_dst, cpu_dst);
        
    }
}
