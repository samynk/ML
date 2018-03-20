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
public class TestSingleLayerDigitRecognition {

    private static final int TRAIN_ITERATIONS = 60000;
    private static final int TEST_ITERATIONS = 2000;
    private static final int BATCHSIZE = 10;
    private static final float LEARNING_RATE = .01f;

    public TestSingleLayerDigitRecognition() {
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

        AbstractLayer layer1 = new Layer(784, 1, 10, BATCHSIZE, ActivationFunction.SOFTMAX);
        layer1.setName("final_layer");
        LearningRate lrd = new LearningRateConst(LEARNING_RATE/BATCHSIZE);
        DeepLayer dl = new DeepLayer(lrd, layer1);
        //dl.setCostFunction(new CrossEntropyCostFunction());
        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -.1f, .1f);

        int maxImage = images.getNrOfColumns();
        fmatrix image = new fmatrix(1, images.getNrOfColumns(), 1, BATCHSIZE);
        fmatrix target = new fmatrix(1, 10, 1, BATCHSIZE);

        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        System.out.println(image.getSizeAsString());
        for (int i = 0; i < TRAIN_ITERATIONS; ++i) {
            long startTime = System.currentTimeMillis();
            target.applyFunction(x -> 0);
            for (int b = 0; b < BATCHSIZE; ++b) {
                int nextImage = r.nextInt(maxImage);
                images.getRow(nextImage, 0, 0, b, image);

                int digit = (int) trainSetLabels.get(0, nextImage);
                target.set(0, digit, 0, b, 1);
                target.makeMaster();
            }
            long endTime = System.currentTimeMillis();
            System.out.println("setting targets took:" + (endTime-startTime));
            image.makeMaster();
            target.makeMaster();

            dl.train(i, image, target, TrainingMode.BATCH);
        }
        dl.sync();
        dl.analyzeWeights();
        dl.writeWeightImages(weightFolder, TRAIN_ITERATIONS);
        DigitRecognitionTester.testDigitRecognition(dl, 1, TEST_ITERATIONS, r);
    }
}
