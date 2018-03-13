/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.Cell;
import dae.matrix.fmatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.cost.CrossEntropyCostFunction;
import dae.neuralnet.io.BinImageReader;
import dae.neuralnet.io.BinLabelReader;
import dae.neuralnet.io.DeepLayerReader;
import dae.neuralnet.io.DeepLayerWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class TestConvolution {

    private static final int TEST_ITERATIONS = 1000;
    private static final int TRAIN_ITERATIONS = 10000;
    private static final int WEIGHT_DEBUG_CYCLE = 1000;
    private static final int BATCH_SIZE = 5;
    private static final float LEARNING_RATE = .01f;

    private String neuralNetBase = "2018_3_9/14_19";

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

    public void testConvolution() {
        // creates a convolution layer with 5 filters with a filter size of 5x5.
        // The input will be interpreted as a 28x28 image.
        // The stride is one and the batch size is also one (only batch size of 1 is supported at the moment).

        //Layer full2 = new Layer(784, 1, 14*14, ActivationFunction.SIGMOID);
        ConvolutionLayer cl1 = new ConvolutionLayer(28, 28, 32, 5, 1, ActivationFunction.RELU);
        cl1.setName("Conv1");
        PoolLayer pl1 = new PoolLayer(28, 28, 32, 2, 2);
        pl1.setName("Pool1");
        ConvolutionLayer cl2 = new ConvolutionLayer(14, 14, 32, 64, 5, 1, ActivationFunction.LEAKYRELU);
        cl2.setName("Conv2");
        PoolLayer pl2 = new PoolLayer(14, 14, 64, 2, 2);
        pl2.setName("Pool2");
        Layer full = new Layer(pl2.getNrOfOutputs(), 1, 10, ActivationFunction.CESIGMOID);
        full.setName("full");
        full.setDropRate(.3f);

        DeepLayer dl = new DeepLayer(new LearningRateConst(LEARNING_RATE), cl1, pl1, cl2, pl2, full);
        dl.setCostFunction(new CrossEntropyCostFunction());

        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -.1f, .1f);

        int maxImage = images.getNrOfColumns();
        fmatrix image = new fmatrix(1, images.getNrOfColumns());
        fmatrix target = new fmatrix(1, 10);

        System.out.println(image.getSizeAsString());
        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        for (int i = 0; i < TRAIN_ITERATIONS; ++i) {
            target.reset();
            for (int b = 0; b < 1; ++b) {
                int nextImage = r.nextInt(maxImage);
                images.getRow(nextImage, b, image);

                int digit = (int) trainSetLabels.get(0, nextImage);
                target.set(b, digit, 1);
            }
            dl.train(i, image, target, TrainingMode.BATCH);
//            if (i == 0) {
//                dl.writeOutputImages();
//            }
            if (i % BATCH_SIZE == 0) {
                dl.adaptWeights(i, BATCH_SIZE);
            }

            if (i % WEIGHT_DEBUG_CYCLE == 0) {
                dl.writeWeightImages(weightFolder, i);
            }
        }

        dl.writeWeightImages(weightFolder, TRAIN_ITERATIONS);
        testDigitRecognition(dl, 1, r);
    }

    
    public void testConvolution2() {
        // creates a convolution layer with 5 filters with a filter size of 5x5.
        // The input will be interpreted as a 28x28 image.
        // The stride is one and the batch size is also one (only batch size of 1 is supported at the moment).

        //Layer full2 = new Layer(784, 1, 14*14, ActivationFunction.SIGMOID);
        ConvolutionLayer cl1 = new ConvolutionLayer(28, 28, 32, 5, 1, ActivationFunction.LEAKYRELU);
        cl1.setName("Conv1");
        PoolLayer pl1 = new PoolLayer(28, 28, 32, 2, 2);
        pl1.setName("Pool1");

        FuzzyficationLayer fl = new FuzzyficationLayer(pl1.getNrOfOutputs(), 10);

        Layer full = new Layer(fl.getNrOfOutputs(), 0, 10, ActivationFunction.CESIGMOID);
        full.setName("full");
        full.setDropRate(.003f);

        DeepLayer dl = new DeepLayer(new LearningRateConst(LEARNING_RATE), cl1, pl1, fl, full);
        dl.setCostFunction(new CrossEntropyCostFunction());

        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -.1f, .1f);

        int maxImage = images.getNrOfColumns();
        fmatrix image = new fmatrix(1, images.getNrOfColumns());
        fmatrix target = new fmatrix(1, 10);

        System.out.println(image.getSizeAsString());
        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        for (int i = 0; i < TRAIN_ITERATIONS; ++i) {
            target.reset();
            for (int b = 0; b < 1; ++b) {
                int nextImage = r.nextInt(maxImage);
                images.getRow(nextImage, b, image);

                int digit = (int) trainSetLabels.get(0, nextImage);
                target.set(b, digit, 1);
            }
            dl.train(i, image, target, TrainingMode.BATCH);
//            if (i == 0) {
//                dl.writeOutputImages();
//            }
            if (i % BATCH_SIZE == 0) {
                dl.adaptWeights(i, BATCH_SIZE);
            }

            if (i % WEIGHT_DEBUG_CYCLE == 0) {
                dl.writeWeightImages(weightFolder, i);
            }
        }
        dl.writeWeightImages(weightFolder, TRAIN_ITERATIONS);

        DeepLayerWriter dlw = new DeepLayerWriter();
        Path export = Paths.get(System.getProperty("user.home"), ".nn", weightFolder, "final.nn");
        dlw.writeDeepLayer(export, dl);
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
            for (int b = 0; b < 1; ++b) {
                int nextImage = r.nextInt(maxImage);
                testImages.getRow(nextImage, b, image);

                int digit = (int) testSetLabels.get(0, nextImage);
                targets[b] = digit;
            }

            dl.forward(image);

            fmatrix result = dl.getLastLayer().getOutputs();
            result.maxPerRow(cs);

            for (int br = 0; br < batchSize; ++br) {
                Cell c = cs.get(br);
                int digit = targets[br];
                if ((c.column) == digit) {
                    success++;
                }
            }

        }
        float succesRate = 100.0f * (success * 1.0f / (TEST_ITERATIONS));
        System.out.println("Number of successes : " + succesRate + "%");
    }

    @Test
    public void testNeuralNet() {
        Path importPath = Paths.get(System.getProperty("user.home"), ".nn", "weights", this.neuralNetBase, "final21.nn");
        DeepLayerReader dr = new DeepLayerReader();
        DeepLayer dl = dr.readDeepLayer(importPath);
        dl.analyzeWeights();
        
        testDigitRecognition(dl);

    }

    private void testDigitRecognition(DeepLayer dl) {
        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        int maxImage = images.getNrOfColumns();
        fmatrix image = new fmatrix(1, images.getNrOfColumns());
        fmatrix target = new fmatrix(1, 10);

        Random r = new java.util.Random(System.currentTimeMillis());
        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        for (int i = 0; i < TRAIN_ITERATIONS; ++i) {
            target.reset();
            for (int b = 0; b < 1; ++b) {
                int nextImage = r.nextInt(maxImage);
                images.getRow(nextImage, b, image);

                int digit = (int) trainSetLabels.get(0, nextImage);
                target.set(b, digit, 1);
            }
            dl.train(i, image, target, TrainingMode.BATCH);
//            if (i == 0) {
//                dl.writeOutputImages();
//            }
            if ((i+1) % BATCH_SIZE == 0) {
                dl.adaptWeights(i, BATCH_SIZE);
            }

            if (i % WEIGHT_DEBUG_CYCLE == 0) {
                dl.writeWeightImages(weightFolder, i);
            }
        }
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
        testDigitRecognition(dl, 1, r);
    }
}
