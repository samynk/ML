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
import org.jocl.cl_mem;

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
        fmatrix mSrc = new fmatrix(5, 5, 1, 2, 2);
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
        fmatrix mSrc2 = new fmatrix(5, 5, 2, 2, 2);
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

        fmatrix mCopy3 = new fmatrix(5, 5, 2, 2, 2);
        GPU.enqueueReadMatrix(mCopy3, mSrc2.getDeviceBuffer().getRMem());
        System.out.println("Zero padded matrix");
        System.out.println(mCopy3);
        assertMatrixEquals(mCopy3, mSrc2);

        // RMatrix
        fmatrix simpleMatrix = new fmatrix(10, 7, 2, 3);
        simpleMatrix.randomize(-1, 1);
        GPU.uploadRMatrix(simpleMatrix);

        fmatrix simpleMatrixCopy = new fmatrix(simpleMatrix);
        simpleMatrix.reset();

        GPU.downloadRMatrix(simpleMatrix);
        assertMatrixEquals(simpleMatrix, simpleMatrixCopy);

        // RWMatrix
        fmatrix rwMatrix = new fmatrix(10, 7, 3, 4);
        rwMatrix.randomize(-1, 1);
        GPU.uploadRWMatrix(rwMatrix);

        fmatrix rwMatrixCopy = new fmatrix(rwMatrix);
        rwMatrix.reset();

        GPU.downloadRWMatrix(rwMatrix);
        assertMatrixEquals(rwMatrixCopy, rwMatrix);
    }

    @Test
    public void testZeroPadding() {
        fmatrix src = new fmatrix(5, 8, 3, 7, 2);
        src.randomize(-2, 2);

        cl_mem mem_src = GPU.uploadRMatrix(src);

        fmatrix dst = new fmatrix(9, 12, 3, 7);
        GPU.enqueueReadMatrix(dst, mem_src);
        
        
    }
}
