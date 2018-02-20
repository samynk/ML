/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.cpu;

import dae.matrix.imatrix;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FMatrixOpCpu {

    /**
     * Calculates the following product : alpha A * B + beta * C, where A*B is a
     * matrix multiplication. The result is stored in C.
     *
     * @param alpha a float value that defines the alpha value.
     * @param A the matrix A.
     * @param B the matrix B.
     * @param beta a float value that defines the beta value.
     * @param C the matrix C, where the result will be stored.
     */
    public void sgemm(float alpha, imatrix A, imatrix B, float beta, imatrix C) {
        C.multiply(beta);

        if (A.getNrOfColumns() != B.getNrOfRows()) {
            Logger.getLogger(FMatrixOpCpu.class.getName()).log(Level.INFO,
                    "Multiply Error , inner dimension must agree: {0} != {1}",
                    new Object[]{A.getSizeAsString(), B.getSizeAsString()});
            return;
        }
        int maxRows = Math.min(A.getNrOfRows(), C.getNrOfRows());
        int maxColumns = Math.min(B.getNrOfColumns(), C.getNrOfColumns());
        for (int c_row = 0; c_row < maxRows; ++c_row) {
            for (int c_column = 0; c_column < maxColumns; ++c_column) {
                float sum = C.get(c_row, c_column);
                for (int index = 0; index < A.getNrOfColumns(); ++index) {
                    sum += A.get(c_row, index) * B.get(index, c_column);
                }
                C.set(c_row, c_column, sum);
            }
        }
    }

    /**
     * Applies a convolution filter on the input matrix.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    public void convolve(imatrix input, imatrix filter, int stride, imatrix output) {
        for (int or = 0; or < output.getNrOfRows(); ++or) {
            for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                float c = convolveSingle(or, oc, input, filter, stride);
                output.set(or, oc, c);
            }
        }
    }

    /**
     * Performs the convolution for a single cell in the output.
     *
     * @param or the current row of the output.
     * @param oc the current column of the output.
     * @param input the input matrix.
     * @param filter the filter.
     * @param stride the stride of the convolution window.
     * @return
     */
    private float convolveSingle(int or, int oc, imatrix input, imatrix filter, int stride) {
        int irb = or * stride;
        int irc = oc * stride;

        float c = 0;
        for (int fr = 0; fr < filter.getNrOfRows(); ++fr) {
            for (int fc = 0; fc < filter.getNrOfColumns(); ++fc) {
                int ir = irb + fr;
                int ic = irc + fc;
                float iv = input.get(ir,ic);
                float fv = filter.get(fr, fc);
                c += iv * fv;
            }
        }
        return c;
    }
}
