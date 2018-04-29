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
public class TestFuzzySingleLayerDR {

    private static final int TRAIN_ITERATIONS = 100000;
    private static final int TEST_ITERATIONS = 20;
    private static final int BATCHSIZE = 50;
    private static final float LEARNING_RATE = .05f;
    
    private String neuralNetBase = "2018_3_22/10_29";

    public TestFuzzySingleLayerDR() {
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

        AbstractLayer layer1 = new Layer(784, 1, 1000, BATCHSIZE, ActivationFunction.RELU);
        layer1.setName("first_layer");

        FuzzyficationLayer layer2 = new FuzzyficationLayer(1000, 5, BATCHSIZE, ActivationFunction.TANH);
        layer2.setName("fuzzy layer");
        
        AbstractLayer layer3 = new Layer(layer2.getNrOfOutputs(), 1, 200, BATCHSIZE, ActivationFunction.RELU);
        layer1.setName("third_layer");
        
        AbstractLayer layer4 = new Layer(200, 1, 10, BATCHSIZE, ActivationFunction.CESIGMOID);
        layer1.setName("final_layer");

        LearningRate lrd = new LearningRateConst(LEARNING_RATE / BATCHSIZE);
        DeepLayer dl = new DeepLayer(lrd, layer1, layer2, layer3, layer4);
        dl.setCostFunction(new CrossEntropyCostFunction());
        Random r = new Random(System.currentTimeMillis());
        dl.randomizeWeights(r, -5f, 5f);

        int maxImage = images.getNrOfHyperSlices();
        fmatrix image = new fmatrix(images.getNrOfRows(), 1, 1, BATCHSIZE);
        fmatrix target = new fmatrix(10, 1, 1, BATCHSIZE);

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
        DigitRecognitionTester.testDigitRecognition(dl, weightFolder, BATCHSIZE, TEST_ITERATIONS, r,true);
    }
    
   
    public void testNeuralNet() {
        Path importPath = Paths.get(System.getProperty("user.home"), ".nn", "weights", this.neuralNetBase, "final.nn");
        DeepLayerReader dr = new DeepLayerReader();
        DeepLayer dl = dr.readDeepLayer(importPath);
        dl.analyzeWeights();
        
        Random r = new java.util.Random();
        String weightFolder = "weights/" + dl.getTrainingStartTimeAsFolder();
        DigitRecognitionTester.testDigitRecognition(dl, weightFolder, BATCHSIZE, TEST_ITERATIONS, r,true);

    }
}
