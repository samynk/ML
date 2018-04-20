/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.Cell;
import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.io.BinImageReader;
import dae.neuralnet.io.BinLabelReader;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DigitRecognitionTester {

    private final static fmatrix TEST_IMAGES;
    private final static fmatrix TEST_SET_LABELS;

    private final static Color WRONG = new Color(0xff8080);
    private final static Color CORRECT = new Color(0x70db70);

    static {
        BinImageReader testbir = new BinImageReader("/data/t10k-images.idx3-ubyte.bin");
        TEST_IMAGES = testbir.getResult();

        BinLabelReader testblr = new BinLabelReader("/data/t10k-labels.idx1-ubyte.bin");
        TEST_SET_LABELS = testblr.getResult();
    }

    public static void testDigitRecognition(DeepLayer dl, String baseFolder, int batchSize, int testIterations, Random r) {
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
        
        boolean randomImage = true;
        if (testIterations == -1){
            testIterations = TEST_IMAGES.getNrOfHyperSlices() / batchSize;
            randomImage = false;
        }
        int nextImage =0;
        for (int i = 0; i < testIterations; ++i) {
            for (int b = 0; b < batchSize; ++b) {
                if ( randomImage ){
                    nextImage = r.nextInt(maxImage);
                }
                TEST_IMAGES.getHyperSlice(nextImage, b, image);
                TEST_SET_LABELS.getHyperSlice(nextImage, b, target);
                ++nextImage;
            }
            image.makeMaster();
            dl.forward(image);

            fmatrix result = (fmatrix)dl.getLastLayer().getOutputs();
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
            writeBatchResults(baseFolder, i, image, result, target);
        }
        float succesRate = 100.0f * (success * 1.0f / (batchSize * testIterations));
        System.out.println("Nr of successes : " + success);
        System.out.println("Success rate : " + succesRate + "%");
    }

    /**
     * Write the results of the batch.
     *
     * @param baseFolder the base folder to write the result of the batch into.
     * @param iteration the iteration of the test.
     * @param images the images of the test.
     * @param results the results of the test.
     * @param target the labels of the test.
     */
    private static void writeBatchResults(String baseFolder, int iteration, fmatrix images, fmatrix results, fmatrix target) {
        String homeDir = System.getProperty("user.home");
        Path exportPath = Paths.get(homeDir, ".nn", baseFolder, "batch" + iteration + ".png");

        BufferedImage bi = new BufferedImage(600, images.getNrOfHyperSlices() * (28 + 3), BufferedImage.TYPE_3BYTE_BGR);
        BufferedImage singleImage = new BufferedImage(28, 28, BufferedImage.TYPE_3BYTE_BGR);
        Graphics g = bi.getGraphics();
        g.setFont(new Font("Arial", Font.PLAIN, 10));
        for (int h = 0; h < images.getNrOfHyperSlices(); ++h) {
            int y = (h * 31) + 3;

            int maxRIndex = -1;
            float maxRValue = -Float.MAX_VALUE;
            int maxTIndex = -1;
            float maxTValue = -Float.MAX_VALUE;
            for (int rr = 0; rr < results.getNrOfRows(); ++rr) {
                float rValue = results.get(rr, 0, 0, h);
                if (rValue > maxRValue) {
                    maxRValue = rValue;
                    maxRIndex = rr;
                }
                String rToDraw = String.format(java.util.Locale.US, "%.3f", rValue);
                g.drawString(rToDraw, 34 + rr * 50, y + 10);

                float tValue = target.get(rr, 0, 0, h);
                if (tValue > maxTValue) {
                    maxTValue = tValue;
                    maxTIndex = rr;
                }
                String tToDraw = String.format(java.util.Locale.US, "%.3f", tValue);
                g.drawString(tToDraw, 34 + rr * 50, y + 20);
            }
            Color c = WRONG;
            if (maxTIndex == maxRIndex) {
                c = CORRECT;
            }
            writeSingleImage(singleImage, c, images, h);
            g.drawImage(singleImage, 3, y, null);
        }

        try {
            Files.createDirectories(exportPath);
            ImageIO.write(bi, "png", exportPath.toFile());

        } catch (IOException ex) {
            Logger.getLogger(Layer.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void writeSingleImage(BufferedImage toWrite, Color color, fmatrix images, int hyperslice) {
        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c) {
                float p = images.get(r * 28 + c, 0, 0, hyperslice);
                int red = (int) (color.getRed() * p);
                int gre = (int) (color.getGreen() * p);
                int blu = (int) (color.getBlue() * p);
                int pi = (int) Math.round(p);

                toWrite.setRGB(c, r, (red << 16) + (gre << 8) + blu);
            }
        }
    }
}
