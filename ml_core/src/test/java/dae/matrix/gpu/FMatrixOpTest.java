/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.cpu.FMatrixOpCpu;
import dae.matrix.fmatrix;
import dae.matrix.integer.intmatrix;
import java.nio.FloatBuffer;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import static dae.matrix.gpu.MatrixTestUtil.*;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FMatrixOpTest {

    private final FMatrixOpCpu cpu = new FMatrixOpCpu();
    private final FMatrixOpGpu gpu = new FMatrixOpGpu();

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

        fmatrix filter = new fmatrix(5, 5);
        filter.applyCellFunction((int row, int column, int slice, float value) -> .1f);
        fmatrix output1 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1);

        long start1 = System.currentTimeMillis();
        gpu.convolve(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("GPU time : " + (end1 - start1));

        fmatrix output2 = new fmatrix(output1.getNrOfRows(), output1.getNrOfColumns());
        long start = System.currentTimeMillis();

        cpu.convolve(input, filter, 1, output2);
        long end = System.currentTimeMillis();

        System.out.println("CPU time : " + (end - start));

        output1.sync();
        assertMatrixEquals(output2, output1);
    }

    @Test
    public void testBatchConvolution() {
        fmatrix input = new fmatrix(100, 100);
        input.randomize(-4, 4);

        int nrOfFilters = 5;

        fmatrix filter = new fmatrix(5, 5, nrOfFilters);
        filter.randomize(-2, 2);
        filter.applyCellFunction((int row, int column, int slice, float value) -> (slice + 1) * .1f);

        fmatrix output1 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters);
        fmatrix output2 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters);

        long start1 = System.currentTimeMillis();
        gpu.batchConvolve(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("Batch convolve - GPU time : " + (end1 - start1));

        long start2 = System.currentTimeMillis();
        cpu.batchConvolve(input, filter, 1, output2);
        long end2 = System.currentTimeMillis();
        System.out.println("Batch convolve - CPU time : " + (end2 - start2));
        output1.sync();
        assertMatrixEquals(output1, output2);
    }

    @Test
    public void testBatchConvolutionWithSlices() {
        fmatrix input = new fmatrix(6, 6, 3);

        input.applyCellFunction((int row, int column, int slice, float value) -> row + column * 5 + slice);

        int nrOfFilters = 2;

        fmatrix filter = new fmatrix(5, 5, nrOfFilters * input.getNrOfSlices());
        filter.randomize(-1, 1);
        //filter.applyCellFunction((int row, int column, int slice, float value) -> (slice + 1) * .1f);

        fmatrix output1 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters * input.getNrOfSlices());
        fmatrix output2 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters * input.getNrOfSlices());

        long start1 = System.currentTimeMillis();
        new FMatrixOpGpu().batchConvolve(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("Batch convolve - GPU time : " + (end1 - start1));

        long start2 = System.currentTimeMillis();
        new FMatrixOpCpu().batchConvolve(input, filter, 1, output2);
        long end2 = System.currentTimeMillis();
        System.out.println("Batch convolve - CPU time : " + (end2 - start2));

//        System.out.println("gpu output");
//        System.out.println(output1);
//        System.out.println("cpu output");
//        System.out.println(output2);
//        assertMatrixEquals(output1, output2);
    }

    /**
     *
     */
    @Test
    public void testConvolutionWithPadding() {

        int nrOfFilters = 3;
        int K = 5;
        // stride is 1
        int S = 1;
        fmatrix filters = new fmatrix(K, K, nrOfFilters);
        filters.applyCellFunction((int row, int column, int slice, float value) -> (slice + 1) * (row + column) * .1f);

//        System.out.println("Filters:");
//        System.out.println(filters);
        // add padding to create output of the same size.
        int P = (K - 1) / 2;
        Random r = new Random();
        int nzpRows = 6;
        int nzpCols = 6;
        fmatrix input = new fmatrix(nzpRows, nzpCols, 1, 1, P);
        input.randomize(-5, 5);

        int O_Col = 1 + (nzpCols - K + 2 * P) / S;
        int O_Row = 1 + (nzpRows - K + 2 * P) / S;

        fmatrix output1 = new fmatrix(O_Row, O_Col, nrOfFilters);
        fmatrix output2 = new fmatrix(O_Row, O_Col, nrOfFilters);

        long start1 = System.currentTimeMillis();
        gpu.batchConvolve(input, filters, S, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("Batch convolve - GPU time : " + (end1 - start1));
        output1.sync();

        long start2 = System.currentTimeMillis();
        cpu.batchConvolve(input, filters, S, output2);
        long end2 = System.currentTimeMillis();
        System.out.println("Batch convolve - CPU time : " + (end2 - start2));

//        System.out.println("input");
//        System.out.println(input);
//
//        System.out.println("output1");
//        System.out.println(output1);
//        System.out.println("output2");
//        System.out.println(output2);
        assertMatrixEquals(output1, output2);
    }

    @Test
    public void testBatchCorrelation() {
        fmatrix input = new fmatrix(7, 7);
        input.applyCellFunction((int row, int column, int slice, float value) -> row + column * 5 + slice);

        int nrOfFilters = 3;

        fmatrix filter = new fmatrix(5, 5, nrOfFilters);
        filter.applyCellFunction((int row, int column, int slice, float value) -> (slice + 1) * .1f);

        fmatrix output1 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters);
        fmatrix output2 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters);

//        System.out.println("Input:");
//        System.out.println(input);
//
//        System.out.println("filter:");
//        System.out.println(filter);
        long start1 = System.currentTimeMillis();
        gpu.batchCorrelate(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("Batch correlate - GPU time : " + (end1 - start1));
        output1.sync();

        long start2 = System.currentTimeMillis();
        cpu.batchCorrelate(input, filter, 1, output2);
        long end2 = System.currentTimeMillis();
        System.out.println("Batch correlate - CPU time : " + (end2 - start2));

//        System.out.println("GPU output:");
//        System.out.println(output1);
//        System.out.println("CPU output");
//        System.out.println(output2);
        assertMatrixEquals(output1, output2);
    }

    @Test
    public void testBatchCorrelationWithSlices() {
        fmatrix input = new fmatrix(6, 6, 3);
        input.applyCellFunction((int row, int column, int slice, float value) -> row + column * 5 + slice);

        int nrOfFilters = 2;

        fmatrix filter = new fmatrix(5, 5, nrOfFilters * input.getNrOfSlices());
        filter.applyCellFunction((int row, int column, int slice, float value) -> (slice + 1) * .1f);

        fmatrix output1 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters * input.getNrOfSlices());
        fmatrix output2 = new fmatrix(input.getNrOfRows() - filter.getNrOfRows() + 1, input.getNrOfColumns() - filter.getNrOfColumns() + 1, nrOfFilters * input.getNrOfSlices());

        long start1 = System.currentTimeMillis();
        gpu.batchCorrelate(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("Batch convolve - GPU time : " + (end1 - start1));
        output1.sync();
        
        long start2 = System.currentTimeMillis();
        cpu.batchCorrelate(input, filter, 1, output2);
        long end2 = System.currentTimeMillis();
        System.out.println("Batch convolve - CPU time : " + (end2 - start2));

//        System.out.println("gpu output");
//        System.out.println(output1);
//        System.out.println("cpu output");
//        System.out.println(output2);
        assertMatrixEquals(output1, output2);
    }

    @Test
    public void testBackpropCorrelation() {
        // 4 input slices of 28x28 with zero padding of 2.
        fmatrix deltas = new fmatrix(28, 28, 4, 1, 2);
        deltas.randomize(-1, 1);
        // 4 filter slices of 5x5 with no zero padding.
        fmatrix filters = new fmatrix(5, 5, 4);
        filters.randomize(-4,4);
        // 2 output slices of 28x28 with no zero padding.
        fmatrix gpu_outputs = new fmatrix(28, 28, 2);
        fmatrix cpu_outputs = new fmatrix(28, 28, 2);

        gpu.batchBackpropCorrelate(deltas, filters, 1, gpu_outputs);
        cpu.batchBackpropCorrelate(deltas, filters, 1, cpu_outputs);
        gpu_outputs.sync();
        
        System.out.println("GPU output");
        System.out.println(gpu_outputs);
        System.out.println("CPU output");
        System.out.println(cpu_outputs);

        assertMatrixEquals(cpu_outputs, gpu_outputs);
    }

    @Test
    public void testSigmoid() {
        fmatrix inputMatrix1 = new fmatrix(4, 12, 3);
        inputMatrix1.randomize(-2, 2);
        fmatrix inputMatrix2 = new fmatrix(inputMatrix1);

        cpu.sigmoid(inputMatrix1);
        gpu.sigmoid(inputMatrix2);

//        System.out.println(inputMatrix1);
//        System.out.println(inputMatrix2);
        inputMatrix2.sync();
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

    @Test
    public void testCopy() {
        fmatrix copy = fmatrix.random(1, 784, -5, +5);
        fmatrix image = new fmatrix(28, 28);

        fmatrix.rowVectorToMatrix(copy, image);

        FloatBuffer source = copy.getHostData();
        FloatBuffer dest = image.getHostData();

        assertArrayEquals(source.array(), dest.array(), 0.0001f);

        fmatrix testCopy = new fmatrix(1, 784);
        fmatrix.matrixToRowVector(image, testCopy);
        assertMatrixEquals(copy, testCopy);

        // with hyper slices
        fmatrix src2 = new fmatrix(1, 784, 1, 2);
        Random r = new Random(System.currentTimeMillis());
        src2.applyFunction(x -> r.nextFloat());
        fmatrix dest2 = new fmatrix(28, 28, 1, 2);

        fmatrix.rowVectorToMatrix(src2, dest2);

        fmatrix dest3 = new fmatrix(1, 784, 1, 2);
        fmatrix.matrixToRowVector(dest2, dest3);

//        System.out.println("comparing ");
//        System.out.println(src2);
//        System.out.println(dest3);
        assertMatrixEquals(src2, dest3);
    }

    @Test
    public void testMaxPool() {
        fmatrix input = new fmatrix(10, 10, 2);
        intmatrix maskLayer1 = new intmatrix(5, 5, 2);
        intmatrix maskLayer2 = new intmatrix(5, 5, 2);
        Random r = new Random(System.currentTimeMillis());
        input.applyFunction(x -> r.nextFloat());

        fmatrix output1 = new fmatrix(5, 5, 2);
        fmatrix output2 = new fmatrix(5, 5, 2);

        cpu.batchMaxPool(input, output1, maskLayer1);
        gpu.batchMaxPool(input, output2, maskLayer2);

        output2.sync();
        maskLayer2.sync();
        assertMatrixEquals(output1, output2);
        assertMatrixEquals(maskLayer1, maskLayer2);

        fmatrix backprop1 = new fmatrix(10, 10, 2);
        fmatrix backprop2 = new fmatrix(10, 10, 2);

        cpu.batchBackpropMaxPool(output1, maskLayer1, backprop1);
        gpu.batchBackpropMaxPool(output1, maskLayer2, backprop2);

        assertMatrixEquals(backprop1, backprop2);
    }

    @Test
    public void testDotSubtract() {
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

        System.out.println("op1");
        System.out.println(op1);
        System.out.println("op2");
        System.out.println(op2);

        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotsubtract(resultCpu, op1, op2);
        gpu.dotsubtract(resultGpu, op1, op2);

        System.out.println("CPU");
        System.out.println(resultCpu);
        System.out.println("GPU");
        System.out.println(resultGpu);

        resultGpu.sync();
        assertMatrixEquals(resultCpu, resultGpu);
    }

    @Test
    public void testDotAdd() {
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

        System.out.println("op1");
        System.out.println(op1);
        System.out.println("op2");
        System.out.println(op2);

        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotadd(resultCpu, op1, op2);
        gpu.dotadd(resultGpu, op1, op2);

        System.out.println("CPU");
        System.out.println(resultCpu);
        System.out.println("GPU");
        System.out.println(resultGpu);
        resultGpu.sync();
        assertMatrixEquals(resultCpu, resultGpu);
    }

    @Test
    public void testDotAddLC() {
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

        System.out.println("op1");
        System.out.println(op1);
        System.out.println("op2");
        System.out.println(op2);

        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotadd(resultCpu, .24f, op1, .7f, op2);
        gpu.dotadd(resultGpu, .24f, op1, .7f, op2);
        resultGpu.sync();

//        System.out.println("CPU");
//        System.out.println(resultCpu);
//        System.out.println("GPU");
//        System.out.println(resultGpu);
        assertMatrixEquals(resultCpu, resultGpu);
    }

    @Test
    public void testDotMultiply() {
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

        System.out.println("op1");
        System.out.println(op1);
        System.out.println("op2");
        System.out.println(op2);

        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotmultiply(resultCpu, op1, op2);
        gpu.dotmultiply(resultGpu, op1, op2);
        resultGpu.sync();

        System.out.println("CPU");
        System.out.println(resultCpu);
        System.out.println("GPU");
        System.out.println(resultGpu);

        assertMatrixEquals(resultCpu, resultGpu);
    }

    @Test
    public void testOperationChain() {
        // test if matrices are uploaded and downloaded as needed.
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotmultiply(resultCpu, op1, op2);
        gpu.dotmultiply(resultGpu, op1, op2);

    }

    @Test
    public void testZeroFill() {
        fmatrix input1 = new fmatrix(100, 107, 5, 2);
        input1.randomize(-5, 5);
        fmatrix inputCopy = new fmatrix(100, 107, 5, 2);

        GPU.zeroFillR(input1);
        GPU.downloadRMatrix(input1);
        assertMatrixEquals(input1, inputCopy);
    }
}
