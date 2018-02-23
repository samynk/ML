package dae.neuralnet;

import dae.matrix.Cell;
import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.io.BinImageReader;
import dae.neuralnet.io.BinLabelReader;
import dae.neuralnet.matrix.MatrixFactory;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class TestLayer {

    private static final int ITERATIONS = 50000;

    public TestLayer() {
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
        Layer l1 = new Layer(2, 1, 2, ActivationFunction.SIGMOID);
        Layer l2 = new Layer(2, 1, 2, ActivationFunction.SIGMOID);

        DeepLayer dl = new DeepLayer(i -> .5f, l1, l2);
        assertTrue(dl.isValid());

        imatrix weights1 = l1.getWeights();
        weights1.setColumn(0, .15f, .20f, .35f);
        weights1.setColumn(1, .25f, .30f, .35f);

        System.out.println(weights1);
        // biases are autmatically set to 1.
        l1.setInputs(.05f, .10f);
        l1.printInputs();
        l1.forward();

        fmatrix output1 = l1.getOutputs();
        System.out.println(output1.toString());

        assertEquals(0.593269992f, output1.get(0, 0), 0.0010f);
        assertEquals(0.596884378f, output1.get(0, 1), 0.0010f);

        imatrix weights2 = l2.getWeights();
        weights2.setColumn(0, .40f, .45f, .60f);
        weights2.setColumn(1, .50f, .55f, .60f);

        l2.setInputs(output1);
        l2.setIdeal(0.01f, 0.99f);
        l2.forward();

        fmatrix output2 = l2.getOutputs();
        assertEquals(0.75136507f, output2.get(0, 0), 0.0010f);
        assertEquals(0.772928465f, output2.get(0, 1), 0.0010f);

        l2.backpropagate(0.5f, true);

        fmatrix errors = l2.getErrors();
        fmatrix E = new fmatrix(errors);
        E.applyFunction(x -> x * x / 2.0f);

        assertEquals(0.274811083f, E.get(0, 0), 0.0010f);
        assertEquals(0.023560026f, E.get(0, 1), 0.0010f);

        float totalError = E.sum();
        assertEquals(0.298371109, totalError, .0001f);

        imatrix weights = l2.getDeltaWeights();
        System.out.println("Weights:");
        System.out.println("________");
        System.out.println(weights);

        float w1 = weights.get(0, 0);
        assertEquals(0.35891648f, w1, 0.001f);
        float w2 = weights.get(0, 1);
        assertEquals(0.511301270f, w2, 0.001f);
        float w3 = weights.get(1, 0);
        assertEquals(0.408666186f, w3, 0.001f);
        float w4 = weights.get(1, 1);
        assertEquals(0.561370121f, w4, 0.001f);

        // calculate the ideals of the first layer.
        fmatrix deltas = l1.getDeltas();
        l2.calculateErrors(deltas);

        System.out.println("deltas for previous layer:");
        System.out.println("__________________________");
        System.out.println(deltas);
        // finally 
        l2.adaptWeights();

        l1.backpropagate(0.5f, false);
        l1.adaptWeights();

        System.out.println(l1.getWeights());

        System.out.println("Test learning");
        fmatrix input = new fmatrix(1, 2);
        input.setRow(0, new float[]{.05f, .10f});

        fmatrix target = new fmatrix(1, 2);
        target.setRow(0, new float[]{0.01f, 0.99f});

        for (int i = 0; i < 1000; ++i) {
            dl.train(i, input, target);
        }
    }

    public void testDigitRecognitionSingleLayer() {
        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        Layer l1 = new Layer(784, 1, 10, ActivationFunction.SOFTMAX);

        int maxImage = images.getNrOfColumns();
        fmatrix image = new fmatrix(1, images.getNrOfColumns());
        fmatrix target = new fmatrix(1, 10);

        Random r = new Random();

        for (int i = 0; i < 100000; ++i) {
            int nextImage = r.nextInt(maxImage);
            images.getRow(nextImage + 1, image);

            int digit = (int) trainSetLabels.get(1, nextImage + 1);

            target.reset();
            target.set(1, digit + 1, 1);

            l1.setInputs(image);
            l1.setIdeal(target);
            l1.forward();
            l1.backpropagate(.1f, true);
            l1.adaptWeights();
        }

        testDigitRecognitionSingleLayer(l1, r);

    }

    private void testDigitRecognitionSingleLayer(Layer l, Random r) {
        // test
        BinImageReader testbir = new BinImageReader("/data/t10k-images.idx3-ubyte.bin");
        fmatrix testImages = testbir.getResult();
        System.out.println(testImages.getSizeAsString());

        BinLabelReader testblr = new BinLabelReader("/data/t10k-labels.idx1-ubyte.bin");
        fmatrix testSetLabels = testblr.getResult();
        System.out.println(testSetLabels.getSizeAsString());

        int maxImage = testImages.getNrOfColumns();
        fmatrix image = new fmatrix(1, testImages.getNrOfColumns());
        fmatrix target = new fmatrix(1, 10);

        System.out.println(image.getSizeAsString());
        int success = 0;
        Cell max = new Cell();
        for (int i = 0; i < 1000; ++i) {
            int nextImage = r.nextInt(maxImage);
            testImages.getRow(nextImage + 1, image);

            int digit = (int) testSetLabels.get(1, nextImage + 1);

            target.reset();
            target.set(1, digit + 1, 1);
            l.setInputs(image);
            l.forward();

            fmatrix result = l.getOutputs();
            Cell c = result.max(max);

            if ((c.column - 1) == digit) {
                success++;
            }
        }
        System.out.println("Number of successes : " + success);
    }

    public void testDigitRecognition() {
        BinImageReader bir = new BinImageReader("/data/train-images.idx3-ubyte.bin");
        fmatrix images = bir.getResult();
        System.out.println(images.getSizeAsString());

        BinLabelReader blr = new BinLabelReader("/data/train-labels.idx1-ubyte.bin");
        fmatrix trainSetLabels = blr.getResult();
        System.out.println(trainSetLabels.getSizeAsString());

        // batch size 1
        Layer l1 = new Layer(784, 1, 784, 1, ActivationFunction.IDENTITY, MatrixFactory.TRANSLATE_MATRIX);
        Layer l2 = new Layer(784, 1, 10, ActivationFunction.SOFTMAX);

        DeepLayer dl = new DeepLayer(i -> .5f, l1, l2);
        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -0.05f, 0.05f);

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

            dl.train(i, image, target);
        }

        testDigitRecognition(dl, r);
    }

    private void testDigitRecognition(DeepLayer dl, Random r) {
        // test
        BinImageReader testbir = new BinImageReader("/data/t10k-images.idx3-ubyte.bin");
        fmatrix testImages = testbir.getResult();
        System.out.println(testImages.getSizeAsString());

        BinLabelReader testblr = new BinLabelReader("/data/t10k-labels.idx1-ubyte.bin");
        fmatrix testSetLabels = testblr.getResult();
        System.out.println(testSetLabels.getSizeAsString());

        int maxImage = testImages.getNrOfColumns();
        fmatrix image = new fmatrix(1, testImages.getNrOfColumns());
        fmatrix target = new fmatrix(1, 10);

        System.out.println(image.getSizeAsString());
        int success = 0;
        for (int i = 0; i < 1000; ++i) {
            int nextImage = r.nextInt(maxImage);
            testImages.getRow(nextImage + 1, image);

            int digit = (int) testSetLabels.get(0, nextImage);

            target.reset();
            target.set(0, digit, 1);

            dl.forward(image);

            fmatrix result = dl.getLastLayer().getOutputs();
            Cell c = result.max();

            if (c.column == digit) {
                success++;
            }
        }
        System.out.println("Number of successes : " + success);
    }

    public void testOneDProblemStandard() {
        // 1 input, 1 bias, 1 output.
        Layer l1 = new Layer(1, 1, 1, ActivationFunction.SIGMOID);

        Random r = new Random();
        r.setSeed(System.currentTimeMillis());
        l1.randomizeWeights(r, -0.05f, 0.05f);

//        System.out.println("Start weights:");
//        System.out.println("______________");
//        System.out.println(l1.getWeights());
        fmatrix input = new fmatrix(1, 1);
        fmatrix target = new fmatrix(1, 1);

        // bigger than 22 degrees is warm, smaller than 22 degrees is cold.
        float cutoff = 22;

        for (int i = 0; i < ITERATIONS; ++i) {
            float T = generateRandom(r, cutoff);

            input.set(1, 1, T);

            if (T > cutoff) {
                target.set(1, 1, 1);
            } else {
                target.set(1, 1, 0);
            }
            l1.setInputs(input);
            l1.setIdeal(target);
            l1.forward();
            l1.backpropagate(1f, true);
            l1.adaptWeights();
        }

        System.out.println("Single layer solution:");
        System.out.println("______________________");
        //System.out.println(l1.getWeights());

        int success = 0;
        for (int i = 0; i < 1000; ++i) {
            float T = generateRandom(r, cutoff);
            input.set(1, 1, T);
            l1.setInputs(input);
            l1.forward();

            fmatrix output = l1.getOutputs();
            float result = output.get(1, 1);

            if (test(result, T, cutoff)) {
                success++;
            }
        }
        System.out.println("Number of successes : " + (success * 1.0f / 1000.0f) * 100 + "%\n");
    }

    public void testOneDProblemSpecial() {
        // 1 input, 1 bias, 1 output.
        Layer l1 = new Layer(1, 1, 1, ActivationFunction.IDENTITY);
        fmatrix constraint = new fmatrix(2, 1);
        constraint.setRow(1, 0);
        l1.setConstraint(constraint);

        Layer l2 = new Layer(1, 0, 1, ActivationFunction.SIGMOID);

        DeepLayer dl = new DeepLayer(i -> 5f, l1, l2);

        Random r = new Random();
        r.setSeed(System.currentTimeMillis());
        dl.randomizeWeights(r, -0.05f, 0.05f);

        // extra constraint
        l1.getWeights().set(0, 0, 1);

        fmatrix input = new fmatrix(1, 1);
        fmatrix target = new fmatrix(1, 1);

        int cutoff = 22;

        for (int i = 0; i < ITERATIONS; ++i) {
            float T = generateRandom(r, cutoff);

            input.set(0, 0, T);

            if (T > cutoff) {
                target.set(0, 0, 1);
            } else {
                target.set(0, 0, 0);
            }
            dl.train(i, input, target);
        }

        System.out.println("With constraint");
        System.out.println("__________________\n");

//        System.out.println("Layer 1 weights:");
//        System.out.println("________________");
//        System.out.println(l1.getWeights());
//
//        System.out.println("Layer 2 weights:");
//        System.out.println("________________");
//        System.out.println(l2.getWeights());
        int success = 0;
        for (int i = 0; i < 1000; ++i) {
            float T = generateRandom(r, cutoff);
            input.set(0, 0, T);
            dl.forward(input);

            fmatrix output = dl.getLastLayer().getOutputs();
            float result = output.get(0, 0);

            if (test(result, T, cutoff)) {
                success++;
            }
        }
        System.out.println("Sucess Rate:" + (success * 1.0f / 1000.0f) * 100 + "%\n");
    }

    private static float generateRandom(Random r, float cutoff) {
        //return 5*((float) r.nextGaussian()) + cutoff;
        return (r.nextFloat() - .5f) * 60 + cutoff;
    }

    private boolean test(float result, float input, float cutoff) {
        return (result > .8f && input > cutoff) || (result < .2f && input < cutoff);
    }

    public void testOneDProblemWithoutConstraint() {
        // 1 input, 1 bias, 1 output.
        Layer l1 = new Layer(1, 1, 1, ActivationFunction.IDENTITY);
        Layer l2 = new Layer(1, 0, 1, ActivationFunction.SIGMOID);

        DeepLayer dl = new DeepLayer(i -> 1f, l1, l2);

        Random r = new Random();
        r.setSeed(System.currentTimeMillis());
        dl.randomizeWeights(r, -0.05f, 0.05f);

        // extra constraint
        l1.getWeights().set(0, 0, 1);

        fmatrix input = new fmatrix(1, 1);
        fmatrix target = new fmatrix(1, 1);

        int cutoff = 22;

        for (int i = 0; i < ITERATIONS; ++i) {
            float T = generateRandom(r, cutoff);
            input.set(0, 0, T);

            if (T > cutoff) {
                target.set(0, 0, 1);
            } else {
                target.set(0, 0, 0);
            }
            dl.train(i, input, target);
        }

        System.out.println("Without constraint");
        System.out.println("__________________\n");

//        System.out.println("Layer 1 weights:");
//        System.out.println("________________");
//        System.out.println(l1.getWeights());
//
//        System.out.println("Layer 2 weights:");
//        System.out.println("________________");
//        System.out.println(l2.getWeights());
        int success = 0;
        for (int i = 0; i < 1000; ++i) {
            float T = generateRandom(r, cutoff);
            input.set(0, 0, T);
            dl.forward(input);

            fmatrix output = dl.getLastLayer().getOutputs();
            float result = output.get(0, 0);

            if (test(result, T, cutoff)) {
                success++;
            }
        }
        System.out.println("Sucess Rate:" + (success * 1.0f / 1000.0f) * 100 + "%\n");
    }

    public void testTranslateLayer() {
        // batch size 1
        Layer l1 = new Layer(2, 1, 2, 1, ActivationFunction.IDENTITY, MatrixFactory.TRANSLATE_MATRIX);
        Layer l2 = new Layer(2, 0, 2, ActivationFunction.SIGMOID);

        DeepLayer dl = new DeepLayer(i -> 1f, l1, l2);

        Random r = new Random();
        r.setSeed(System.currentTimeMillis());
        dl.randomizeWeights(r, -0.05f, 0.05f);

        fmatrix input = new fmatrix(1, 2);
        fmatrix target = new fmatrix(1, 2);

        int cutoff1 = 22; // temperature
        int cutoff2 = 40; // humidity 

        for (int i = 0; i < ITERATIONS; ++i) {
            float T = generateRandom(r, cutoff1);
            float H = generateRandom(r, cutoff2);
            input.set(0, 0, T);
            input.set(0, 1, H);
            target.set(0, 0, T > cutoff1 ? 1 : 0);
            target.set(0, 1, H > cutoff2 ? 1 : 0);

            dl.train(i, input, target);
        }

        int success = 0;
        for (int i = 0; i < 1000; ++i) {
            float T = generateRandom(r, cutoff1);
            float H = generateRandom(r, cutoff2);
            input.set(0, 0, T);
            input.set(0, 1, H);
            dl.forward(input);

            fmatrix output = dl.getLastLayer().getOutputs();
            float result1 = output.get(0, 0);
            float result2 = output.get(0, 1);

            if (test(result1, T, cutoff1) && test(result2, H, cutoff2)) {
                success++;
            }
        }
        System.out.println("TRANSLATE LAYER TEST");
        System.out.println("____________________\n");
        System.out.println("Sucess Rate:" + (success * 1.0f / 1000.0f) * 100 + "%\n");
    }
}
