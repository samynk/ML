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
import dae.neuralnet.activation.ActivationFunction;
import java.nio.file.Paths;

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
        gpu.batchConvolve(input, filter, 1, output1);
        long end1 = System.currentTimeMillis();
        System.out.println("Batch convolve - GPU time : " + (end1 - start1));

        long start2 = System.currentTimeMillis();
        cpu.batchConvolve(input, filter, 1, output2);
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
        filters.randomize(-4, 4);
        // 2 output slices of 28x28 with no zero padding.
        fmatrix gpu_outputs = new fmatrix(28, 28, 2);
        fmatrix cpu_outputs = new fmatrix(28, 28, 2);

        gpu.batchBackpropCorrelate(deltas, filters, 1, gpu_outputs);
        cpu.batchBackpropCorrelate(deltas, filters, 1, cpu_outputs);
        gpu_outputs.sync();

//        System.out.println("GPU output");
//        System.out.println(gpu_outputs);
//        System.out.println("CPU output");
//        System.out.println(cpu_outputs);
        assertMatrixEquals(cpu_outputs, gpu_outputs);
    }

    @Test
    public void testActivation() {
        testActivationFunction(ActivationFunction.SIGMOID);
        testActivationFunction(ActivationFunction.CESIGMOID);
        testActivationFunction(ActivationFunction.RELU);
        testActivationFunction(ActivationFunction.LEAKYRELU);
        testActivationFunction(ActivationFunction.TANH);

        testDActivationFunction(ActivationFunction.SIGMOID);
        testDActivationFunction(ActivationFunction.CESIGMOID);
        testDActivationFunction(ActivationFunction.IDENTITY);
        testDActivationFunction(ActivationFunction.RELU);
        testDActivationFunction(ActivationFunction.LEAKYRELU);
        testDActivationFunction(ActivationFunction.TANH);
    }

    private void testActivationFunction(ActivationFunction function) {
        Random r = new Random(System.currentTimeMillis());
        int rs = r.nextInt(10) + 5;
        int cs = r.nextInt(10) + 5;
        int ss = r.nextInt(10) + 5;
        int hs = r.nextInt(10) + 4;

        fmatrix aGPU = new fmatrix(rs, cs, ss, hs);
        aGPU.randomize(-5, 5);
        fmatrix aCPU = new fmatrix(aGPU);
        cpu.applyActivation(function, aCPU);
        gpu.applyActivation(function, aGPU);
        aGPU.sync();

        assertMatrixEquals(aGPU, aCPU);
    }

    private void testDActivationFunction(ActivationFunction function) {
        Random r = new Random(System.currentTimeMillis());
        int rs = r.nextInt(10) + 5;
        int cs = r.nextInt(10) + 5;
        int ss = r.nextInt(10) + 5;
        int hs = r.nextInt(10) + 4;

        fmatrix reluGPU = new fmatrix(rs, cs, ss, hs);
        reluGPU.randomize(-5, 5);
        fmatrix reluCPU = new fmatrix(reluGPU);
        cpu.applyDerivedActivation(function, reluCPU);
        gpu.applyDerivedActivation(function, reluGPU);
        reluGPU.sync();

        assertMatrixEquals(reluGPU, reluCPU);
    }

    @Test
    public void testMatrixMultiply() {
        fmatrix op1 = fmatrix.random(23, 7, -5, 10);
        fmatrix op2 = fmatrix.random(7, 15, -10, 7);
        fmatrix result1 = fmatrix.zeros(23, 15);
        fmatrix result2 = fmatrix.zeros(23, 15);

        this.gpu.sgemm(1, op1, op2, 0, result1);
        this.cpu.sgemm(1, op1, op2, 0, result2);

        fmatrix op3 = new fmatrix(15, 32);
        op3.randomize(-5, 5);

        fmatrix result2cpu = new fmatrix(23, 32);
        fmatrix result2gpu = new fmatrix(23, 32);
        this.cpu.sgemm(1, result2, op3, .5f, result2cpu);
        this.gpu.sgemm(1, result1, op3, .5f, result2gpu);

        result1.sync();
        assertMatrixEquals(result1, result2);

        result2gpu.sync();
        assertMatrixEquals(result2cpu, result2gpu);
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

        fmatrix src3 = new fmatrix(16, 1, 1, 10);
        src3.randomize(-5, 5);
        // zeropad 2
        fmatrix dest4 = new fmatrix(4, 4, 1, 10, 2);
        fmatrix.copyIntoSlice(src3, dest4);
        dest4.sync();

        FloatBuffer src3Buffer = src3.getHostData();
        FloatBuffer dst4Buffer = dest4.getHostData();

        assertArrayEquals(src3Buffer.array(), dst4Buffer.array(), 0.0001f);

    }

    @Test
    public void testMaxPool() {
        int rows = 20;
        int columns = 20;
        int slices = 10;
        int batchSize = 50;

        fmatrix input = new fmatrix(rows, columns, slices, batchSize);
        input.randomize(-3, 3);
        intmatrix maskLayer1 = new intmatrix(rows / 2, columns / 2, slices, batchSize);
        intmatrix maskLayer2 = new intmatrix(rows / 2, columns / 2, slices, batchSize);

        fmatrix output1 = new fmatrix(rows / 2, columns / 2, slices, batchSize);
        fmatrix output2 = new fmatrix(rows / 2, columns / 2, slices, batchSize);

        cpu.batchMaxPool(input, output1, maskLayer1);
        gpu.batchMaxPool(input, output2, maskLayer2);

        output2.sync();
        maskLayer2.sync();
        assertMatrixEquals(output1, output2);
        assertMatrixEquals(maskLayer1, maskLayer2);

        fmatrix backprop1 = new fmatrix(rows, columns, slices, batchSize);
        fmatrix backprop2 = new fmatrix(rows, columns, slices, batchSize);

        cpu.batchBackpropMaxPool(output1, maskLayer1, 2, 2, backprop1);
        gpu.batchBackpropMaxPool(output1, maskLayer2, 2, 2, backprop2);

        backprop2.sync();
        assertMatrixEquals(backprop1, backprop2);
    }

    @Test
    public void testDotSubtract() {
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

//        System.out.println("op1");
//        System.out.println(op1);
//        System.out.println("op2");
//        System.out.println(op2);
        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotsubtract(resultCpu, op1, op2);
        gpu.dotsubtract(resultGpu, op1, op2);

//        System.out.println("CPU");
//        System.out.println(resultCpu);
//        System.out.println("GPU");
//        System.out.println(resultGpu);
        resultGpu.sync();
        assertMatrixEquals(resultCpu, resultGpu);
    }

    @Test
    public void testDotAdd() {
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

//        System.out.println("op1");
//        System.out.println(op1);
//        System.out.println("op2");
//        System.out.println(op2);
        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotadd(resultCpu, op1, op2);
        gpu.dotadd(resultGpu, op1, op2);

//        System.out.println("CPU");
//        System.out.println(resultCpu);
//        System.out.println("GPU");
//        System.out.println(resultGpu);
        resultGpu.sync();
        assertMatrixEquals(resultCpu, resultGpu);
    }

    @Test
    public void testDotAddLC() {
        fmatrix op1 = new fmatrix(5, 7, 3);
        op1.randomize(-5, 5);
        fmatrix op2 = new fmatrix(5, 7, 3);
        op2.randomize(-5, 5);

//        System.out.println("op1");
//        System.out.println(op1);
//        System.out.println("op2");
//        System.out.println(op2);
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

//        System.out.println("op1");
//        System.out.println(op1);
//        System.out.println("op2");
//        System.out.println(op2);
        fmatrix resultCpu = new fmatrix(5, 7, 3);
        fmatrix resultGpu = new fmatrix(5, 7, 3);

        cpu.dotmultiply(resultCpu, op1, op2);
        gpu.dotmultiply(resultGpu, op1, op2);
        resultGpu.sync();

//        System.out.println("CPU");
//        System.out.println(resultCpu);
//        System.out.println("GPU");
//        System.out.println(resultGpu);
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

        GPU.zeroFill(input1);
        GPU.download(input1);
        assertMatrixEquals(input1, inputCopy);
    }

    @Test
    public void testDotSquare() {
        fmatrix input1Gpu = new fmatrix(400, 200, 5, 2);
        input1Gpu.randomize(-10, 10);

        fmatrix input1Cpu = new fmatrix(input1Gpu);

        cpu.square(input1Cpu);
        gpu.square(input1Gpu);
        input1Gpu.sync();
        assertMatrixEquals(input1Cpu, input1Gpu);
    }

    @Test
    public void testAdamVelocity() {
        fmatrix input1Gpu = new fmatrix(400, 200, 5, 2);
        input1Gpu.randomize(-10, 10);
        fmatrix input1Cpu = new fmatrix(input1Gpu);

        fmatrix input2Gpu = new fmatrix(400, 200, 5, 2);
        input2Gpu.randomize(-10, 10);
        fmatrix input2Cpu = new fmatrix(input2Gpu);

        cpu.adamVelocity(input1Cpu, 0.9f, input1Cpu, input2Cpu);
        gpu.adamVelocity(input1Gpu, 0.9f, input1Gpu, input2Gpu);
        input1Gpu.sync();
        assertMatrixEquals(input1Cpu, input1Gpu);
    }

    @Test
    public void testAdamAdaptWeights() {
        int r = 10;
        int c = 7;

        fmatrix gpuW = new fmatrix(r, c, 5, 2);
        fmatrix gpuM = new fmatrix(r, c, 5, 2);
        fmatrix gpuV = new fmatrix(r, c, 5, 2);
        gpuW.randomize(-10, 10);
        gpuM.randomize(-10, 10);
        // gradient can not be negative.
        gpuV.randomize(1, 10);

        fmatrix cpuW = new fmatrix(gpuW);
        fmatrix cpuM = new fmatrix(gpuM);
        fmatrix cpuV = new fmatrix(gpuV);

        cpu.adamAdaptWeights(cpuW, 0.1f, 0.9f, 0.999f, 1e-4f, cpuM, cpuV);
        gpu.adamAdaptWeights(gpuW, 0.1f, 0.9f, 0.999f, 1e-4f, gpuM, gpuV);
        gpuW.sync();
        assertMatrixEquals(cpuW, gpuW);
    }

    @Test
    public void testDotMultiplyFactor() {
        fmatrix input1Gpu = new fmatrix(400, 200, 5, 2);
        input1Gpu.randomize(-10, 10);
        fmatrix input1Cpu = new fmatrix(input1Gpu);

        cpu.dotmultiply(input1Cpu, input1Cpu, 0.5542f);
        gpu.dotmultiply(input1Gpu, input1Gpu, 0.5542f);
        input1Gpu.sync();
        assertMatrixEquals(input1Cpu, input1Gpu);
    }

    @Test
    public void testRandom() {
        fmatrix toRandomize = new fmatrix(10, 10);
        gpu.randomize(toRandomize, -5, 5);
    }

    @Test
    public void testRotateKernel() {
        fmatrix kernel = new fmatrix(49, 49, 4);
        // set pattern
        for (int x = 0; x < 49; ++x) {
            for (int y = 0; y < 49; ++y) {
                float val = (y / 7 + x / 7) % 2 == 0 ? 1 : 0;
                for (int fs = 0; fs < 4; ++fs) {
                    switch (fs) {
                        case 0:
                        case 1:
                        case 2:
                            if (val > 0.01f) {
                                val = val - fs * 0.05f;
                            }
                            break;
                        case 3:
                            val = 0.5f;
                            break;
                    }

                    kernel.set(y, x, fs, val);
                }
            }
        }
        fmatrix.writeAs3DImage(kernel, 4, 10, Paths.get("startRotation"));

        fmatrix outputGpu = new fmatrix(49, 49, 28);
        fmatrix outputCpu = new fmatrix(49, 49, 28);
        gpu.rotateKernels(kernel, 7, 0, (float) (Math.PI - Math.PI / 8.0), outputGpu);
        cpu.rotateKernels(kernel, 7, 0, (float) (Math.PI - Math.PI / 8.0), outputCpu);

        outputGpu.sync();
        fmatrix.writeAs3DImage(outputCpu, 7, 10, Paths.get("rotationCpu"));
        fmatrix.writeAs3DImage(outputGpu, 7, 10, Paths.get("rotationGpu"));

        assertMatrixEquals(outputCpu, outputGpu);

    }

    @Test
    public void testAccumulateRotateKernel() {

        fmatrix kernel = new fmatrix(49, 49, 28);
        kernel.randomize(-1, 1);
        fmatrix.writeAs3DImage(kernel, 4, 10, Paths.get("startRotation"));

        fmatrix outputGpu = new fmatrix(49, 49, 4);
        fmatrix outputCpu = new fmatrix(49, 49, 4);
        gpu.accumulateRotateKernels(kernel, 7, 0, (float) (Math.PI - Math.PI / 8.0), outputGpu);
        cpu.accumulateRotateKernels(kernel, 7, 0, (float) (Math.PI - Math.PI / 8.0), outputCpu);

        outputGpu.sync();
        fmatrix.writeAs3DImage(outputCpu, 7, 10, Paths.get("rotationCpu"));
        fmatrix.writeAs3DImage(outputGpu, 7, 10, Paths.get("rotationGpu"));

        assertMatrixEquals(outputCpu, outputGpu);

    }

    @Test
    public void maxRotation() {
        int rows = 28;
        int columns = 28;
        int features = 6;
        int rotations = 7;
        int batchSize = 100;

        fmatrix input = new fmatrix(rows, columns, features * rotations, batchSize);
        input.randomize(-1, 1);

        fmatrix outputCpu = new fmatrix(rows, columns, features, batchSize);
        fmatrix rotOutputCpu = new fmatrix(rows, columns, features, batchSize);
        fmatrix outputGpu = new fmatrix(rows, columns, features, batchSize);
        fmatrix rotOutputGpu = new fmatrix(rows, columns, features, batchSize);

        gpu.maxRotation(input, features, rotations, 0, (float) Math.PI, outputGpu, rotOutputGpu);
        outputGpu.sync();
        rotOutputGpu.sync();
        cpu.maxRotation(input, features, rotations, 0, (float) Math.PI, outputCpu, rotOutputCpu);

        assertMatrixEquals(outputCpu, outputGpu);
        assertMatrixEquals(rotOutputCpu, rotOutputGpu);

        fmatrix output2Cpu = new fmatrix(rows, columns, features * rotations, batchSize);
        fmatrix output2Gpu = new fmatrix(rows, columns, features * rotations, batchSize);
        gpu.maxInverseRotation(outputGpu, rotOutputGpu, features, rotations, 0, (float) Math.PI, output2Gpu);
        cpu.maxInverseRotation(outputCpu, rotOutputCpu, features, rotations, 0, (float) Math.PI, output2Cpu);
        output2Gpu.sync();

        System.out.println("Cpu");
        //System.out.println(output2Cpu);
        System.out.println("Gpu");
        //System.out.println(output2Gpu);

        assertMatrixEquals(output2Cpu, output2Gpu);
    }

    @Test
    public void testZip() {
        fmatrix input1 = new fmatrix(14, 14, 24, 100);
        input1.randomize(-1, 1);

        fmatrix input2 = new fmatrix(14, 14, 24, 100);
        input2.randomize(-1, 1);

        fmatrix gpuDest = new fmatrix(14, 14, 48, 100);
        fmatrix cpuDest = new fmatrix(14, 14, 48, 100);
        gpu.zip(input1, input2, gpuDest);
        cpu.zip(input1, input2, cpuDest);

        gpuDest.sync();

        assertMatrixEquals(gpuDest, cpuDest);

        fmatrix cpuInput1 = new fmatrix(14, 14, 24, 100);
        fmatrix cpuInput2 = new fmatrix(14, 14, 24, 100);
        cpu.unzip(cpuDest, cpuInput1, cpuInput2);

        fmatrix gpuInput1 = new fmatrix(14, 14, 24, 100);
        fmatrix gpuInput2 = new fmatrix(14, 14, 24, 100);
        gpu.unzip(gpuDest, gpuInput1, gpuInput2);

        assertMatrixEquals(input1, cpuInput1);
        assertMatrixEquals(input2, cpuInput2);

        gpuInput1.sync();
        gpuInput2.sync();
        assertMatrixEquals(input1, gpuInput1);
        assertMatrixEquals(input2, gpuInput2);

    }

    @Test
    public void testMaxRotationPool() {
        int rows = 20;
        int columns = 20;
        int slices = 8;
        int batchSize = 50;

        fmatrix src = new fmatrix(rows, columns, slices, batchSize);
        src.randomize(-1, 1);

        fmatrix gpuDest = new fmatrix(rows / 2, columns / 2, slices, batchSize);
        intmatrix gpuMask = new intmatrix(rows / 2, columns / 2, slices / 2, batchSize);

        fmatrix cpuDest = new fmatrix(rows / 2, columns / 2, slices, batchSize);
        intmatrix cpuMask = new intmatrix(rows / 2, columns / 2, slices / 2, batchSize);

        gpu.maxRotationPool(src, gpuDest, gpuMask);
        cpu.maxRotationPool(src, cpuDest, cpuMask);

        gpuDest.sync();
        gpuMask.sync();

        assertMatrixEquals(cpuMask, gpuMask);
        assertMatrixEquals(cpuDest, gpuDest);

        fmatrix gpuBP = new fmatrix(rows, columns, slices, batchSize);
        fmatrix cpuBP = new fmatrix(rows, columns, slices, batchSize);

        gpu.backpropMaxRotationPool(gpuDest, gpuMask, gpuBP);
        cpu.backpropMaxRotationPool(cpuDest, cpuMask, cpuBP);

        gpuBP.sync();
        assertMatrixEquals(cpuBP, gpuBP);
    }
    
    @Test 
    public void testPancake(){
        int rows = 20;
        int columns = 20;
        int slices = 20;
        int batchSize = 50;
        
        int slicesPerGroup = 4;
        int biases = 1;
        
        fmatrix input = new fmatrix(rows,columns,slices, batchSize);
        int weightSlices = slicesPerGroup*(biases+slices / slicesPerGroup);
        fmatrix weights = new fmatrix(rows, columns, weightSlices, batchSize);
        
        fmatrix outputGPU = new fmatrix(rows, columns, slices/slicesPerGroup, batchSize);
        fmatrix outputCPU = new fmatrix(rows, columns, slices/slicesPerGroup, batchSize);
        
        cpu.forwardPancake(input, slicesPerGroup, biases, weights, outputCPU);
        gpu.forwardPancake(input, slicesPerGroup, biases, weights, outputGPU);
        
        outputGPU.sync();
        assertMatrixEquals(outputCPU, outputGPU);
    }
}
