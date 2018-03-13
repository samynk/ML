/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.fmatrix;
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

    @Test
    public void testFuzzyLayer() {
        FuzzyficationLayer single = new FuzzyficationLayer(1, 4);
        assertEquals(single.getNrOfOutputs(), 4);
        Random r = new Random(System.currentTimeMillis());
        single.randomizeWeights(r, 0, 100);
        fmatrix input = new fmatrix(1, 1);
        
        

        System.out.println(single.getBWeights());
        for (float x = -50; x < 0.0f; ++x) {
            input.set(0,0, x);
            single.setInputs(input);
            single.forward();
            fmatrix outputs = single.getOutputs();
            System.out.println("x:" + x);
            System.out.println(outputs);
        }
    }
}
