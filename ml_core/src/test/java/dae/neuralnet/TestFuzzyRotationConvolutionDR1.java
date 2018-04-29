/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.Dimension;
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

    private static final int TRAIN_ITERATIONS = 10000;
    private static final int EPOCHTEST = 500;
    private static final int EPOCHTESTITERATION = 10;
    private static final int TEST_ITERATIONS = -1;
    private static final int BATCHSIZE = 40;
    private static final float LEARNING_RATE = 0.1f;

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

        int numStartFilters = 8;

        RotationConvolutionLayer layer1 = new RotationConvolutionLayer(28, 28, numStartFilters, 8, 5, 1, BATCHSIZE, ActivationFunction.RELU);
        layer1.setName("convolution_layer 1");

        RotationPoolLayer rpl = new RotationPoolLayer(28, 28, numStartFilters, 8, BATCHSIZE);
        rpl.setName("Rotation pool layer");

        MaxRotationPoolLayer mpl = new MaxRotationPoolLayer(28, 28, numStartFilters * 2, 2, 2, BATCHSIZE);
        mpl.setName("max rotation pool");

        int mplOutputs = mpl.getNrOfOutputs();

        RotationConvolutionLayer vl2 = new RotationConvolutionLayer(14, 14, numStartFilters, 4, 8, 5, 1, BATCHSIZE, ActivationFunction.RELU);
        vl2.setAngles(0, (float) Math.PI / 2);
        RotationConvolutionLayer rl2 = new RotationConvolutionLayer(14, 14, numStartFilters, 4, 8, 5, 1, BATCHSIZE, ActivationFunction.RELU);
        rl2.setAngles(0, (float) Math.PI / 2);
        rl2.setAddAngleBias(true);
        CompositeLayer cl = new CompositeLayer(
                Dimension.Dim(14, 14, numStartFilters * 2, BATCHSIZE),
                Dimension.Dim(14, 14, numStartFilters * 64, BATCHSIZE),
                vl2, rl2);
        cl.setName("Composite layer");

        PancakeLayer pcl1 = new PancakeLayer(
                Dimension.Dim(14, 14, numStartFilters * 64, BATCHSIZE),
                true, 2, ActivationFunction.IDENTITY);

        RotationPoolLayer rpl2 = new RotationPoolLayer(14, 14, numStartFilters * 4, 8, BATCHSIZE);
        rpl2.setAngles(0, (float) Math.PI / 2);

        MaxRotationPoolLayer mpl2 = new MaxRotationPoolLayer(14, 14, numStartFilters * 8, 2, 2, BATCHSIZE);
        AbstractLayer layer6 = new Layer(mpl2.getNrOfOutputs(), 1, 60, BATCHSIZE, ActivationFunction.LEAKYRELU);
        layer1.setName("final_layer");

        AbstractLayer layer7 = new Layer(layer6.getNrOfOutputs(), 1, 1, BATCHSIZE, ActivationFunction.CESIGMOID);
        layer1.setName("final_layer");

        DemuxLayer demux = new DemuxLayer(10, BATCHSIZE, layer6, layer7);
        demux.setName("demux");

//        AbstractLayer layer8 = new Layer(layer7.getNrOfOutputs(), 1, 10, BATCHSIZE, ActivationFunction.CESIGMOID);
//        layer8.setName("layer8");
        LearningRate lrd = new LearningRateConst(LEARNING_RATE / BATCHSIZE);
        //DeepLayer dl = new DeepLayer(lrd, layer1, rpl, mpl, cl, pcl1, rpl2, mpl2, layer6, layer7);
        DeepLayer dl = new DeepLayer(lrd, layer1, rpl, mpl, cl, pcl1, rpl2, mpl2, demux);

        dl.setCostFunction(new CrossEntropyCostFunction());
        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -1, 1f);

        int maxImage = images.getNrOfHyperSlices();
        fmatrix image = new fmatrix(images.getNrOfRows(), 1, 1, BATCHSIZE);
        fmatrix target = new fmatrix(10, 1, 1, BATCHSIZE);

        image.setName("Image");
        target.setName("target");

        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        String outputFolder = "outputs/" + dl.getTrainingStartTimeAsFolder();
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

            if (i % EPOCHTEST == 0) {
                //layer6.enableDropLayer(false);
                String epochFolder = weightFolder + "/epoch" + (1 + i / EPOCHTEST);
                DigitRecognitionTester.testDigitRecognition(dl, epochFolder, BATCHSIZE, EPOCHTESTITERATION, r, false);
                //layer6.enableDropLayer(true);
            }
        }

        DeepLayerWriter dlw = new DeepLayerWriter();
        Path origin = null;
        boolean increment = false;
        if (dl.getMetaData().getPath() != null) {
            origin = dl.getMetaData().getPath();
            increment = true;
        } else {
            origin = Paths.get(System.getProperty("user.home"), ".nn", weightFolder, "final.nn");
        }
        //layer6.enableDropLayer(false);
        dlw.writeDeepLayer(origin, increment, dl);
        DigitRecognitionTester.testDigitRecognition(dl, weightFolder, BATCHSIZE, TEST_ITERATIONS, r, true);

        dl.sync();
        dl.analyzeWeights();
        dl.writeWeightImages(weightFolder, TRAIN_ITERATIONS);
        dl.writeOutputImages(outputFolder);
    }

    public void testNeuralNet() {
        Path importPath = Paths.get(System.getProperty("user.home"), ".nn", "weights", this.neuralNetBase, "final.nn");
        DeepLayerReader dr = new DeepLayerReader();
        DeepLayer dl = dr.readDeepLayer(importPath);
        dl.analyzeWeights();

        Random r = new java.util.Random();
        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        DigitRecognitionTester.testDigitRecognition(dl, weightFolder, BATCHSIZE, TEST_ITERATIONS, r, true);

    }
}
