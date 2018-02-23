/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.cpu.FMatrixOpCpu;
import dae.matrix.fmatrix;
import dae.matrix.fsubmatrix;
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

    private FMatrixOpCpu cpu = new FMatrixOpCpu();
    private FMatrixOpGpu gpu = new FMatrixOpGpu();

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
        fmatrix input = new fmatrix(1800, 1800);
        input.applyCellFunction((int row, int column, int slice, float value) -> (row + column * 5));

        fsubmatrix sm = new fsubmatrix(input, 0, 0, 5, 5);
        System.out.println("Input : ");
        System.out.println(sm);

        fmatrix filter = new fmatrix(5, 5);
        filter.applyCellFunction((int row, int column, int slice, float value) -> .1f);
        fmatrix output1 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1);

        long start1 = System.currentTimeMillis();
        FMatrixOpGpu gpu = new FMatrixOpGpu();
        gpu.convolve(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("GPU time : " + (end1 - start1));

        fmatrix output2 = new fmatrix(output1.getNrOfRows(), output1.getNrOfColumns());
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

    @Test
    public void testBatchConvolution() {
        fmatrix input = new fmatrix(100, 100);
        input.applyCellFunction((int row, int column, int slice, float value) -> row + column * 5 + slice);

        int nrOfFilters = 5;

        fmatrix filter = new fmatrix(5, 5, nrOfFilters);
        filter.applyCellFunction((int row, int column, int slice, float value) -> (slice + 1) * .1f);

        fmatrix output1 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters);
        fmatrix output2 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters);

        long start1 = System.currentTimeMillis();
        new FMatrixOpGpu().batchConvolve(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("Batch convolve - GPU time : " + (end1 - start1));

        long start2 = System.currentTimeMillis();
        new FMatrixOpCpu().batchConvolve(input, filter, 1, output2);
        long end2 = System.currentTimeMillis();
        System.out.println("Batch convolve - CPU time : " + (end2 - start2));

        assertMatrixEquals(output1, output2);
    }

    private void assertMatrixEquals(fmatrix output1, fmatrix output2) {
        for (int s = 0; s < output1.getNrOfSlices(); ++s) {
            for (int r = 0; r < output1.getNrOfRows(); ++r) {
                for (int c = 0; c < output1.getNrOfColumns(); ++c) {
                    float v1 = output1.get(r, c, s);
                    float v2 = output2.get(r, c, s);
                    assertEquals("error on :" + r + "," + c, v1, v2, 0.01f);
                }
            }
        }
    }

    @Test
    public void testSigmoid() {
        fmatrix inputMatrix1 = fmatrix.random(5, 5, -1.0f, 1.0f);
        fmatrix inputMatrix2 = new fmatrix(inputMatrix1);

        cpu.sigmoid(inputMatrix1);
        gpu.sigmoid(inputMatrix2);

        System.out.println(inputMatrix1);
        System.out.println(inputMatrix2);

        assertMatrixEquals(inputMatrix1, inputMatrix2);
    }

    @Test
    public void testMatrixMultiply() {
        fmatrix op1 = fmatrix.random(23, 7, -5, 10);
        fmatrix op2 = fmatrix.random(7, 15, -10, 7);
        fmatrix result1 = fmatrix.zeros(23, 15);
        fmatrix result2 = fmatrix.zeros(23, 15);

        this.gpu.sgemm(1, op1, op2, 0, result1);
        this.cpu.sgemm(1, op1, op2, 0, result2);

        assertMatrixEquals(result1, result2);
    }
}
