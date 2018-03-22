/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.io;

import dae.matrix.gpu.MatrixTestUtil;
import dae.matrix.imatrix;
import dae.neuralnet.ConvolutionLayer;
import dae.neuralnet.DeepLayer;
import dae.neuralnet.FuzzyficationLayer;
import dae.neuralnet.ILayer;
import dae.neuralnet.Layer;
import dae.neuralnet.LearningRateConst;
import dae.neuralnet.PoolLayer;
import dae.neuralnet.activation.ActivationFunction;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Calendar;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
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
public class DeepLayerRWTest {

    public DeepLayerRWTest() {
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
    public void testDeepLayerRW() {
        try {
            Random r = new java.util.Random(System.currentTimeMillis());
            Layer original = new Layer(100, 1, 36, ActivationFunction.IDENTITY);
            original.setName("firstlayer");
            original.randomizeWeights(r, -2, 2);

            ConvolutionLayer oc = new ConvolutionLayer(6, 6, 1, 5, 5, 1, 5, ActivationFunction.RELU);
            oc.setName("convolution1");
            oc.randomizeWeights(r, -5, 5);

            PoolLayer op = new PoolLayer(6, 6, 5, 2, 2, 5);
            op.setName("maxpool1");

            FuzzyficationLayer fl = new FuzzyficationLayer(op.getNrOfOutputs(), 5, 1);
            fl.setName("fuzzy1");
            fl.randomizeWeights(r, .1f, 50);

            DeepLayer dl = new DeepLayer(new LearningRateConst(.1f), original, oc, op, fl);
            dl.getMetaData().setAuthor("Koen Samyn");
            DeepLayerWriter dlw = new DeepLayerWriter();

            Path file = Files.createTempFile("dae", ".nn");
            dlw.writeDeepLayer(file, dl);

            DeepLayerReader dlr = new DeepLayerReader();
            DeepLayer dl2 = dlr.readDeepLayer(file);

            assertNotNull(dl2);
            DeepLayerMetaData dlmd = dl2.getMetaData();
            assertNotNull(dlmd);
            Calendar createTime = dlmd.getCreationTime();

            Calendar current = Calendar.getInstance();
            assertEquals(createTime.get(Calendar.YEAR), current.get(Calendar.YEAR));
            assertEquals(createTime.get(Calendar.MONTH), current.get(Calendar.MONTH));
            assertEquals(createTime.get(Calendar.DAY_OF_MONTH), current.get(Calendar.DAY_OF_MONTH));

            assertEquals("Koen Samyn", dlmd.getAuthor());

            assertEquals(dl.getLearningRate(), dl2.getLearningRate());
            assertEquals(dl.getCostFunction(), dl2.getCostFunction());

            assertEquals(dl.getNrOfLayers(), dl2.getNrOfLayers());

            // first layer
            ILayer layer = dl2.getFirstLayer();
            assertNotNull(layer);
            assertTrue(layer instanceof Layer);
            Layer l = (Layer) layer;
            assertLayerEquals(original, l);

            // second layer
            ILayer layer2 = dl2.getLayer(1);
            assertNotNull(layer2);
            assertTrue(layer2 instanceof ConvolutionLayer);
            ConvolutionLayer rc = (ConvolutionLayer) layer2;
            assertLayerEquals(oc, rc);

            // third layer
            ILayer layer3 = dl2.getLayer(2);
            assertNotNull(layer3);
            assertTrue(layer3 instanceof PoolLayer);
            PoolLayer rp = (PoolLayer) layer3;
            assertLayerEquals(op, rp);

            // fourth layer
            ILayer layer4 = dl2.getLayer(3);
            assertNotNull(layer4);
            assertTrue(layer4 instanceof FuzzyficationLayer);
            FuzzyficationLayer fr = (FuzzyficationLayer) layer4;
            assertLayerEquals(fl, fr);

        } catch (IOException ex) {
            Logger.getLogger(DeepLayerRWTest.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    private void assertLayerEquals(ConvolutionLayer oc, ConvolutionLayer rc) {
        assertEquals(oc.getName(), rc.getName());
        assertEquals(oc.getNrOfWInputs(), rc.getNrOfWInputs());
        assertEquals(oc.getNrOfHInputs(), rc.getNrOfHInputs());
        assertEquals(oc.getNrOfSInputs(), rc.getNrOfSInputs());
        assertEquals(oc.getNrOfFeatures(), rc.getNrOfFeatures());
        assertEquals(oc.getFilterSize(), rc.getFilterSize());
        assertEquals(oc.getFilterStride(), rc.getFilterStride());
        assertEquals(oc.getActivationFunction(), rc.getActivationFunction());
        MatrixTestUtil.assertMatrixEquals(oc.getWeights(), rc.getWeights());
    }

    private void assertLayerEquals(Layer original, Layer l) {
        assertEquals(original.getName(), l.getName());
        assertEquals(original.getNrOfInputs(), l.getNrOfInputs());
        assertEquals(original.getNrOfBiases(), l.getNrOfBiases());
        assertEquals(original.getNrOfOutputs(), l.getNrOfOutputs());
        assertEquals(original.getActivationFunction(), l.getActivationFunction());
        imatrix oWeights = original.getWeights();
        imatrix rWeights = l.getWeights();
        MatrixTestUtil.assertMatrixEquals(oWeights, rWeights);
    }

    private void assertLayerEquals(PoolLayer oc, PoolLayer rc) {
        assertEquals(oc.getName(), rc.getName());
        assertEquals(oc.getNrofWInputs(), rc.getNrofWInputs());
        assertEquals(oc.getNrOfHInputs(), rc.getNrOfHInputs());
        assertEquals(oc.getNrOfSInputs(), rc.getNrOfSInputs());
        assertEquals(oc.getScaleX(), rc.getScaleX());
        assertEquals(oc.getScaleY(), rc.getScaleY());
    }

    private void assertLayerEquals(FuzzyficationLayer original, FuzzyficationLayer l) {
        assertEquals(original.getName(), l.getName());
        assertEquals(original.getNrOfInputs(), l.getNrOfInputs());
        assertEquals(original.getNrOfOutputs(), l.getNrOfOutputs());
        assertEquals(original.getActivationFunction(), l.getActivationFunction());
        MatrixTestUtil.assertMatrixEquals(original.getAWeights(), l.getAWeights());
        MatrixTestUtil.assertMatrixEquals(original.getBWeights(), l.getBWeights());

    }
}
