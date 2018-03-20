/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.Cell;
import dae.matrix.fmatrix;
import dae.neuralnet.io.BinImageReader;
import dae.neuralnet.io.BinLabelReader;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DigitRecognitionTester {

    private static fmatrix TEST_IMAGES;
    private static fmatrix TEST_SET_LABELS;

    static {
        BinImageReader testbir = new BinImageReader("/data/t10k-images.idx3-ubyte.bin");
        TEST_IMAGES = testbir.getResult();

        BinLabelReader testblr = new BinLabelReader("/data/t10k-labels.idx1-ubyte.bin");
        TEST_SET_LABELS = testblr.getResult();
    }

    public static void testDigitRecognition(DeepLayer dl, int batchSize, int testIterations, Random r) {
        // test
        int maxImage = TEST_IMAGES.getNrOfHyperSlices();
        fmatrix image = new fmatrix(TEST_IMAGES.getNrOfRows(), 1, 1, batchSize);

        System.out.println(image.getSizeAsString());
        int success = 0;
        ArrayList<Cell> cs = new ArrayList<>();
        ArrayList<Cell> ts = new ArrayList<>();
        for (int i = 0; i < batchSize; ++i) {
            cs.add(new Cell());
            ts.add(new Cell());
        }
        fmatrix target = new fmatrix(10, 1, 1, batchSize);
        for (int i = 0; i < testIterations; ++i) {
            for (int b = 0; b < batchSize; ++b) {
                int nextImage = r.nextInt(maxImage);
                TEST_IMAGES.getHyperSlice(nextImage, b, image);
                TEST_SET_LABELS.getHyperSlice(nextImage, b, target);
            }
            image.makeMaster();
            dl.forward(image);

            fmatrix result = dl.getLastLayer().getOutputs();
            result.sync();
            result.maxPerColumn(cs);
            target.maxPerColumn(ts);

            for (int br = 0; br < batchSize; ++br) {
                Cell c = cs.get(br);
                Cell t = ts.get(br);
                if (c.row == t.row) {
                    success++;
                }
            }
            System.out.println(result);
            System.out.println(target);

        }
        float succesRate = 100.0f * (success * 1.0f / (batchSize * testIterations));
        System.out.println("Number of successes : " + succesRate + "%");
    }
}
