/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.cpu.FMatrixOpCpu;
import dae.matrix.fmatrix;
import dae.matrix.gpu.FMatrixOpGpu;
import dae.matrix.gpu.MatrixTestUtil;
import static dae.matrix.gpu.MatrixTestUtil.assertMatrixEquals;
import java.util.Random;
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
public class TestFuzzyLayer {

    private final FMatrixOpCpu cpu = new FMatrixOpCpu();
    private final FMatrixOpGpu gpu = new FMatrixOpGpu();

    public TestFuzzyLayer() {
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

    public void testFuzzyLayer() {
        FuzzyficationLayer single = new FuzzyficationLayer(1, 4, 10);
        assertEquals(single.getNrOfOutputs(), 4);
        Random r = new Random(System.currentTimeMillis());
        single.randomizeWeights(r, 0, 100);
        fmatrix input = new fmatrix(1, 1);

        System.out.println(single.getBWeights());
        for (float x = -50; x < 0.0f; ++x) {
            input.set(0, 0, x);
            single.setInputs(input);
            single.forward();
            fmatrix outputs = single.getOutputs();
            System.out.println("x:" + x);
            System.out.println(outputs);
        }
    }

    @Test
    public void testFuzzyFunction() {
        int nrOfInputs = 51;
        int nrOfClasses = 4;
        int batchSize = 30;

        fmatrix inputGPU = new fmatrix(nrOfInputs, 1, 1, batchSize);
        inputGPU.randomize(-3, +3);
        fmatrix a = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1);
        a.randomize(-1, 1);
        fmatrix b = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1);
        b.randomize(-1, 1);
        fmatrix outputGPU = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1, 1, batchSize);
        fmatrix outputCPU = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1, 1, batchSize);

        fmatrix inputCPU = new fmatrix(inputGPU);

        gpu.fuzzyFunction(inputGPU, nrOfClasses, a, b, outputGPU);
        cpu.fuzzyFunction(inputCPU, nrOfClasses, a, b, outputCPU);

        outputGPU.sync();
        MatrixTestUtil.assertMatrixEquals(outputCPU, outputGPU);
    }

    @Test
    public void testFuzzyOneMinus() {
        int nrOfInputs = 51;
        int nrOfClasses = 4;
        int batchSize = 30;

        fmatrix inputGPU = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1, 1, batchSize);
        inputGPU.randomize(-3, +3);
        fmatrix inputCPU = new fmatrix(inputGPU);

        fmatrix outputGPU = new fmatrix(nrOfInputs * nrOfClasses, 1, 1, batchSize);
        fmatrix outputCPU = new fmatrix(nrOfInputs * nrOfClasses, 1, 1, batchSize);

        gpu.fuzzyShiftMinus(inputGPU, nrOfClasses, outputGPU);
        cpu.fuzzyShiftMinus(inputCPU, nrOfClasses, outputCPU);

        outputGPU.sync();
        MatrixTestUtil.assertMatrixEquals(outputCPU, outputGPU);
    }

    @Test
    public void testFuzzyShiftDeltas() {
        int nrOfInputs = 51;
        int nrOfClasses = 4;
        int batchSize = 30;

        fmatrix inputGPU = new fmatrix(nrOfInputs * nrOfClasses, 1, 1, batchSize);
        inputGPU.randomize(-3, +3);
        fmatrix inputCPU = new fmatrix(inputGPU);

        fmatrix outputGPU = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1, 1, batchSize);
        fmatrix outputCPU = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1, 1, batchSize);

        gpu.fuzzyShiftDeltas(inputGPU, nrOfClasses, outputGPU);
        cpu.fuzzyShiftDeltas(inputCPU, nrOfClasses, outputCPU);

        outputGPU.sync();
        MatrixTestUtil.assertMatrixEquals(outputCPU, outputGPU);
    }

    @Test
    public void testSumPerRow() {
        fmatrix inputGPU = new fmatrix(1000, 1, 1, 30);
        inputGPU.randomize(-4f, 4f);
        fmatrix outputGPU = new fmatrix(1000, 1, 1, 1);
        fmatrix outputCPU = new fmatrix(1000, 1, 1, 1);

        gpu.sumPerRow(inputGPU, outputGPU);
        cpu.sumPerRow(inputGPU, outputCPU);
        outputGPU.sync();
        System.out.println(inputGPU);
        assertMatrixEquals(outputCPU, outputGPU);
    }

    @Test
    public void testFuzzyBackProp() {
        int nrOfInputs = 51;
        int nrOfClasses = 4;
        int batchSize = 30;

        fmatrix inputGPU = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1, 1, batchSize);
        inputGPU.randomize(-3, +3);
        fmatrix inputCPU = new fmatrix(inputGPU);

        fmatrix weights = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1);

        fmatrix outputGPU = new fmatrix(nrOfInputs, 1, 1, batchSize);
        fmatrix outputCPU = new fmatrix(nrOfInputs, 1, 1, batchSize);

        cpu.fuzzyBackProp(inputGPU, weights, nrOfClasses, outputGPU);
        gpu.fuzzyBackProp(inputCPU, weights, nrOfClasses, outputCPU);

        outputGPU.sync();
        assertMatrixEquals(outputCPU, outputGPU);

    }
    
    @Test
    public void testFuzzyInputAdd(){
        int nrOfInputs = 51;
        int nrOfClasses = 4;
        int batchSize = 30;

        fmatrix inputGPU = new fmatrix(nrOfInputs, 1, 1, batchSize);
        inputGPU.randomize(-3, +3);
        
        fmatrix b = new fmatrix(nrOfInputs * (nrOfClasses - 1), 1);
        b.randomize(-1, 1);
        
        fmatrix outputGPU = new fmatrix(nrOfInputs*(nrOfClasses-1),1,1,batchSize);
        fmatrix outputCPU = new fmatrix(nrOfInputs*(nrOfClasses-1),1,1,batchSize);
        
        cpu.fuzzyInputAdd(inputGPU, b, nrOfClasses, outputCPU);
        gpu.fuzzyInputAdd(inputGPU, b, nrOfClasses, outputGPU);
        
        outputGPU.sync();
        assertMatrixEquals(outputCPU, outputGPU);
    }
}
