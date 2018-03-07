/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.fmatrix;
import dae.matrix.integer.intmatrix;
import static org.junit.Assert.assertEquals;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class MatrixTestUtil {
     public static void assertMatrixEquals(fmatrix output1, fmatrix output2) {
        for (int s = 0; s < output1.getNrOfSlices(); ++s) {
            for (int r = 0; r < output1.getNrOfRows(); ++r) {
                for (int c = 0; c < output1.getNrOfColumns(); ++c) {
                    float v1 = output1.get(r, c, s);
                    float v2 = output2.get(r, c, s);
                    assertEquals("error on :" + r + "," + c + "," + s, v1, v2, 0.01f);
                }
            }
        }
    }

    public static void assertMatrixEquals(intmatrix output1, intmatrix output2) {
        for (int s = 0; s < output1.getNrOfSlices(); ++s) {
            for (int r = 0; r < output1.getNrOfRows(); ++r) {
                for (int c = 0; c < output1.getNrOfColumns(); ++c) {
                    int v1 = output1.get(r, c, s);
                    int v2 = output2.get(r, c, s);
                    assertEquals("error on :" + r + "," + c + "," + s, v1, v2, 0.01f);
                }
            }
        }
    }
}
