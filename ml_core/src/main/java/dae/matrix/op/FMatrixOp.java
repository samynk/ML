/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.op;

import dae.matrix.imatrix;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public interface FMatrixOp {

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
    public void sgemm(float alpha, imatrix A, imatrix B, float beta, imatrix C);

    /**
     * Applies a convolution filter on the input matrix.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    public void convolve(imatrix input, imatrix filter, int stride, imatrix output);

    /**
     * Applies a convolution filter on the input matrix, with the slices taken
     * into account.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    public void batchConvolve(imatrix input, imatrix filter, int stride, imatrix output);
    /**
     * Applies a correlation filter on the input matrix, with the slices taken
     * into account.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    public void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output);

    /**
     * Calculates the sigmoid activation function. The result is stored back
     * into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void sigmoid(imatrix O);

    /**
     * Calculates the derivative of the sigmoid activation function. The result
     * is stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void dsigmoid(imatrix O);
}
