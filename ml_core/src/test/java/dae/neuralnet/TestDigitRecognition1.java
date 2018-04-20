package dae.neuralnet;

import dae.matrix.Cell;
import dae.matrix.fmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.io.BinImageReader;
import dae.neuralnet.io.BinLabelReader;
import java.util.ArrayList;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class TestDigitRecognition1 {

    private static final int TEST_ITERATIONS = 100;
    private static final int TRAIN_ITERATIONS = 500;
    private static final int BATCH_SIZE = 50;
    private static final float LEARNING_RATE = 0.001f;

    public TestDigitRecognition1() {
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

    public void testDigitRecognition() {
        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        AbstractLayer l1 = new TranslateLayer(784, ActivationFunction.IDENTITY);
        AbstractLayer l2 = new Layer(784, 1, 800, ActivationFunction.SIGMOID);
        AbstractLayer l3 = new Layer(800, 0, 10, ActivationFunction.SOFTMAX);

        LearningRateDecay lrd = new LearningRateDecay(.5f, 0001f);
        DeepLayer dl = new DeepLayer(lrd, l1, l2, l3);
        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -0.001f, 0.001f);

        int maxImage = images.getNrOfColumns();
        fmatrix image = new fmatrix(1, images.getNrOfColumns());
        fmatrix target = new fmatrix(1, 10);

        System.out.println(image.getSizeAsString());
        for (int i = 0; i < 100000; ++i) {
            int nextImage = r.nextInt(maxImage);
            images.getRow(nextImage, image);

            int digit = (int) trainSetLabels.get(0, nextImage);

            target.reset();
            target.set(0, digit, 1);

            dl.train(i, image, target, TrainingMode.STOCHASTIC);
        }

        testDigitRecognition(dl, 1, r);
    }

    private void testDigitRecognition(DeepLayer dl, int batchSize, Random r) {
        // test
        BinImageReader testbir = new BinImageReader("/data/t10k-images.idx3-ubyte.bin");
        fmatrix testImages = testbir.getResult();
        System.out.println(testImages.getSizeAsString());

        BinLabelReader testblr = new BinLabelReader("/data/t10k-labels.idx1-ubyte.bin");
        fmatrix testSetLabels = testblr.getResult();
        System.out.println(testSetLabels.getSizeAsString());

        int maxImage = testImages.getNrOfColumns();

        fmatrix image = new fmatrix(batchSize, testImages.getNrOfColumns());

        System.out.println(image.getSizeAsString());
        int success = 0;
        ArrayList<Cell> cs = new ArrayList<>();
        for (int i = 0; i < batchSize; ++i) {
            cs.add(new Cell());
        }
        int targets[] = new int[batchSize];
        for (int i = 0; i < TEST_ITERATIONS; ++i) {
            for (int b = 0; b < batchSize; ++b) {
                int nextImage = r.nextInt(maxImage);
                testImages.getRow(nextImage, b, image);

                int digit = (int) testSetLabels.get(0, nextImage);
                targets[b] = digit;
            }

            dl.forward(image);

            fmatrix result = (fmatrix)dl.getLastLayer().getOutputs();
            result.maxPerRow(cs);

            for (int br = 0; br < batchSize; ++br) {
                Cell c = cs.get(br);
                int digit = targets[br];
                if ((c.column) == digit) {
                    success++;
                }
            }

        }
        float succesRate = 100.0f * (success * 1.0f / (BATCH_SIZE * TEST_ITERATIONS));
        System.out.println("Number of successes : " + succesRate + "%");
    }

    @Test
    public void testDigitRecognitionBatch() {
        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        //AbstractLayer l1 = new Layer(784, 0, 784, BATCH_SIZE, ActivationFunction.SIGMOID);
        AbstractLayer l1 = new Layer(784, 0, 400, BATCH_SIZE, ActivationFunction.LEAKYRELU);
        AbstractLayer l2 = new Layer(400, 0, 200, BATCH_SIZE, ActivationFunction.LEAKYRELU);
        AbstractLayer l3 = new Layer(200, 0, 100, BATCH_SIZE, ActivationFunction.SIGMOID);
        AbstractLayer l4 = new Layer(100, 0, 50, BATCH_SIZE, ActivationFunction.LEAKYRELU);
        AbstractLayer l5 = new Layer(50, 0, 25, BATCH_SIZE, ActivationFunction.SIGMOID);
        AbstractLayer l6 = new Layer(25, 0, 10, BATCH_SIZE, ActivationFunction.SOFTMAX);

        LearningRate lrd = new LearningRateDecay(LEARNING_RATE, .0001f);
        DeepLayer dl = new DeepLayer(lrd, l1, l2, l3, l4, l5, l6);

        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -0.1f, 0.1f);

        int maxImage = images.getNrOfColumns();
        fmatrix image = new fmatrix(BATCH_SIZE, images.getNrOfColumns());
        fmatrix target = new fmatrix(BATCH_SIZE, 10);

        System.out.println(image.getSizeAsString());
        for (int i = 0; i < TRAIN_ITERATIONS; ++i) {
            target.reset();
            for (int b = 0; b < BATCH_SIZE; ++b) {
                int nextImage = r.nextInt(maxImage);
                images.getRow(nextImage, b, image);

                int digit = (int) trainSetLabels.get(0, nextImage);
                target.set(b, digit, 1);
            }
            dl.train(i, image, target, TrainingMode.BATCH);
        }
        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        dl.writeWeightImages(weightFolder, TRAIN_ITERATIONS);
        testDigitRecognition(dl, BATCH_SIZE, r);
    }

}
