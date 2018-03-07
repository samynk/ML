/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.fmatrix;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import static dae.matrix.gpu.MatrixTestUtil.*;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FMatrixUploadTest {

    public FMatrixUploadTest() {
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
    public void testReadAndWrite() {
        // padding of two with one slice
        fmatrix mSrc = new fmatrix(5, 5, 1, 2);
        mSrc.randomize(-1, 1);
        GPU.uploadRMatrix(mSrc);

        fmatrix mCopy = new fmatrix(mSrc);
        mSrc.reset();
        System.out.println("Before download");
        System.out.println(mSrc);
        
        GPU.downloadRMatrix(mSrc);
        System.out.println("src");
        System.out.println(mSrc);
        System.out.println("copy");
        System.out.println(mCopy);
        assertMatrixEquals(mSrc, mCopy);
        
        // padding of two with two slices
        
        fmatrix mSrc2 = new fmatrix(5, 5, 2, 2);
        mSrc2.randomize(-1, 1);
        GPU.uploadRMatrix(mSrc2);

        fmatrix mCopy2 = new fmatrix(mSrc2);
        mSrc2.reset();

        GPU.downloadRMatrix(mSrc2);
        System.out.println("src");
        System.out.println(mSrc2);
        System.out.println("copy");
        System.out.println(mCopy2);
        assertMatrixEquals(mSrc2, mCopy2);
        
        fmatrix mCopy3 = new fmatrix(9,9,2);
        GPU.enqueueReadMatrix(mCopy3, mSrc2.getDeviceBuffer().getCLReadMem());
        System.out.println("Zero padded matrix");
        System.out.println(mCopy3);

    }
}
