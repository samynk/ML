/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.neuralnet.activation.ActivationFunction;
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
public class TestConvolution {

    public TestConvolution() {
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
        // creates a convolution layer with 5 filters with a filter size of 5x5.
        // The input will be interpreted as a 28x28 image.
        // The stride is one and the batch size is also one (only batch size of 1 is supported at the moment).
        ConvolutionLayer layer = new ConvolutionLayer(5, 28, 28, 0, 5, 1, 1, ActivationFunction.IDENTITY);
        
        
    }
}
