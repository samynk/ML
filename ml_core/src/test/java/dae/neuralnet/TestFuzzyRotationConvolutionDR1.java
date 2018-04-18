/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.cost.CrossEntropyCostFunction;
import dae.neuralnet.io.BinImageReader;
import dae.neuralnet.io.BinLabelReader;
import dae.neuralnet.io.DeepLayerReader;
import dae.neuralnet.io.DeepLayerWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
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
public class TestFuzzyRotationConvolutionDR1 {

    private static final int TRAIN_ITERATIONS = 100000;
    private static final int TEST_ITERATIONS = -1;
    private static final int BATCHSIZE = 100;
    private static final float LEARNING_RATE = 0.05f;

    private String neuralNetBase = "2018_3_22/10_29";

    public TestFuzzyRotationConvolutionDR1() {
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
    public void testSingleLayer() {
        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        int numStartFilters = 4;

        RotationConvolutionLayer layer1 = new RotationConvolutionLayer(28, 28, numStartFilters, 16, 5, 1, BATCHSIZE, ActivationFunction.LEAKYRELU);
        layer1.setName("convolution_layer 1");

        RotationPoolLayer rpl = new RotationPoolLayer(28, 28, numStartFilters, 16, BATCHSIZE);

        MaxRotationPoolLayer mpl = new MaxRotationPoolLayer(28, 28, numStartFilters * 2, 2, 2, BATCHSIZE);

        ConvolutionLayer convLayer2 = new ConvolutionLayer(14, 14, numStartFilters * 2, 4, 5, 1, BATCHSIZE, ActivationFunction.LEAKYRELU);
        layer1.setName("convolution_layer 2");

        PoolLayer pl2 = new PoolLayer(14, 14, numStartFilters * 8, 2, 2, BATCHSIZE);
        pl2.setName("pool layer 2");

        ConvolutionLayer convLayer3 = new ConvolutionLayer(7, 7, numStartFilters * 8, 4, 3, 1, BATCHSIZE, ActivationFunction.LEAKYRELU);
        layer1.setName("convolution_layer 3");

        AbstractLayer layer6 = new Layer(convLayer3.getNrOfOutputs(), 1, 1000, BATCHSIZE, ActivationFunction.RELU);
        layer6.setName("final_layer");

        AbstractLayer layer7 = new Layer(layer6.getNrOfOutputs(), 1, 10, BATCHSIZE, ActivationFunction.CESIGMOID);
        layer7.setName("final_layer");

        LearningRate lrd = new LearningRateConst(LEARNING_RATE / BATCHSIZE);
        DeepLayer dl = new DeepLayer(lrd, layer1, rpl, mpl, convLayer2, pl2, convLayer3, layer6, layer7);

        dl.setCostFunction(new CrossEntropyCostFunction());
        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -5f, 5f);

        int maxImage = images.getNrOfHyperSlices();
        fmatrix image = new fmatrix(images.getNrOfRows(), 1, 1, BATCHSIZE);
        fmatrix target = new fmatrix(10, 1, 1, BATCHSIZE);

        image.setName("Image");
        target.setName("target");

        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        System.out.println(image.getSizeAsString());
        for (int i = 0; i < TRAIN_ITERATIONS; ++i) {
            for (int b = 0; b < BATCHSIZE; ++b) {
                int nextImage = r.nextInt(maxImage);
                images.getHyperSlice(nextImage, b, image);
                trainSetLabels.getHyperSlice(nextImage, b, target);
            }
            image.makeMaster();
            target.makeMaster();

            dl.train(i, image, target, TrainingMode.BATCH);
        }
        dl.sync();
        dl.analyzeWeights();
        dl.writeWeightImages(weightFolder, TRAIN_ITERATIONS);

        DeepLayerWriter dlw = new DeepLayerWriter();
        Path origin = null;
        boolean increment = false;
        if (dl.getMetaData().getPath() != null) {
            origin = dl.getMetaData().getPath();
            increment = true;
        } else {
            origin = Paths.get(System.getProperty("user.home"), ".nn", weightFolder, "final.nn");
        }
        dlw.writeDeepLayer(origin, increment, dl);
        DigitRecognitionTester.testDigitRecognition(dl, weightFolder, BATCHSIZE, TEST_ITERATIONS, r);
    }

    public void testNeuralNet() {
        Path importPath = Paths.get(System.getProperty("user.home"), ".nn", "weights", this.neuralNetBase, "final.nn");
        DeepLayerReader dr = new DeepLayerReader();
        DeepLayer dl = dr.readDeepLayer(importPath);
        dl.analyzeWeights();

        Random r = new java.util.Random();
        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        DigitRecognitionTester.testDigitRecognition(dl, weightFolder, BATCHSIZE, TEST_ITERATIONS, r);

    }
}
