package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import dae.matrix.op.FMatrixOp;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clEnqueueWriteBuffer;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import org.jocl.Sizeof;
import static org.jocl.blast.CLBlast.CLBlastSgemm;
import static org.jocl.blast.CLBlastLayout.CLBlastLayoutColMajor;
import static org.jocl.blast.CLBlastTranspose.*;
import org.jocl.cl_event;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class FMatrixOpGpu implements FMatrixOp {

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
    @Override
    public void sgemm(float alpha, imatrix A, imatrix B, float beta, imatrix C) {
        // Create the device input buffers
        int M = A.getNrOfRows();
        int K = B.getNrOfRows();
        int N = C.getNrOfColumns();

        DeviceBuffer ADB = A.getDeviceBuffer();
        DeviceBuffer BDB = B.getDeviceBuffer();
        DeviceBuffer CDB = C.getDeviceBuffer();
        cl_mem memA = ADB.getCLReadMem();
        cl_mem memB = BDB.getCLReadMem();
        cl_mem memC = CDB.getCLReadWriteMem();

        // Copy the host data to the device
        clEnqueueWriteBuffer(GPU.CL_COMMAND_QUEUE, memA, CL_TRUE, 0, M * K
                * Sizeof.cl_float, ADB.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(GPU.CL_COMMAND_QUEUE, memB, CL_TRUE, 0, K * N
                * Sizeof.cl_float, BDB.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(GPU.CL_COMMAND_QUEUE, memC, CL_TRUE, 0, M * N
                * Sizeof.cl_float, CDB.getCLPointer(), 0, null, null);

        int lda = A.getNrOfRows();
        if (A.isTransposed()) {
            lda = A.getNrOfColumns();
        }

        int ldb = B.getNrOfRows();
        if (B.isTransposed()) {
            ldb = B.getNrOfColumns();
        }
        // Execute GEMM:
        // C = alpha * A * B + beta * C
        cl_event event = new cl_event();
        CLBlastSgemm(CLBlastLayoutColMajor,
                A.isTransposed() ? CLBlastTransposeYes : CLBlastTransposeNo,
                B.isTransposed() ? CLBlastTransposeYes : CLBlastTransposeNo,
                M, N, K,
                alpha,
                memA, 0, lda,
                memB, 0, ldb,
                beta,
                memC, 0, M,
                GPU.CL_COMMAND_QUEUE, event);

        clEnqueueReadBuffer(GPU.CL_COMMAND_QUEUE, memC, CL_TRUE, 0, M * N
                * Sizeof.cl_float, CDB.getCLPointer(), 0, null, null);
    }

    /**
     * Applies a convolution filter on the input matrix.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    @Override
    public void convolve(imatrix input, imatrix filter, int stride, imatrix output) {
        GPU.KERNEL_CONVOLV.convolv(input, filter, output);
    }

    /**
     * Applies a convolution filter on the input matrix, with the slices taken
     * into account.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    @Override
    public void batchConvolve(imatrix input, imatrix filter, int stride, imatrix output) {
        GPU.KERNEL_CONVOLV.batchConvolv(input, filter, stride, output);
    }

    /**
     * Applies a correlation filter on the input matrix, with the slices taken
     * into account.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    @Override
    public void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        GPU.KERNEL_CONVOLV.batchCorrelate(input, filter, stride, output);
    }

    /**
     * Performs the back propagation correlation operation in batch.
     *
     * @param input The input matrix.
     * @param filter The filter matrix.
     * @param stride the stride
     * @param output The output matrix.
     */
    @Override
    public void batchBackpropCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        GPU.KERNEL_CONVOLV.batchBackpropCorrelate(input, filter, stride, output);
    }

    /**
     * Calculates the sigmoid activation function. The result is stored back
     * into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    @Override
    public void sigmoid(imatrix O) {
        GPU.KERNEL_ACTIVATION.sigmoid(O);
    }

    /**
     * Applies a max pool on the input matrix and stores it into the output
     * matrix. It is assumed that the dimensions of the output matrix are
     * dividers of the dimensions of the input matrix.
     *
     * The resulting maskLayer can be used to back propagate deltas to the
     * previous layer. The maskLayer will be filled with zeros and the location
     * of the maximum per filter will be set to 1.
     *
     * @param input the input matrix.
     * @param output the output matrix.
     * @param maskLayer a matrix with the same dimension as the input layer
     * which can be used to determine which input pixels contributed to the
     * output.
     */
    @Override
    public void batchMaxPool(imatrix input, imatrix output, intmatrix maskLayer) {
        GPU.KERNEL_POOL.maxPool(input, output, maskLayer);
    }

    /**
     * Scales up the input matrix to the dimensions of the output matrix. Only
     * the cells that are defined in the masking layer are applied to output.
     *
     * @param input the input matrix.
     * @param maskLayer a matrix with the same dimension as the input layer
     * which can be used to determine which input pixels contributed to the
     * output.
     * @param output the output matrix.
     *
     */
    @Override
    public void batchBackpropMaxPool(imatrix input, intmatrix maskLayer, imatrix output) {
        GPU.KERNEL_POOL.backpropMaxPool(input, maskLayer, output);
    }

    /**
     * Calculates the derivative of the sigmoid activation function. The result
     * is stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    @Override
    public void dsigmoid(imatrix O) {
        GPU.KERNEL_ACTIVATION.dsigmoid(O);
    }

    /**
     * Calculates the element by element addition of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    public imatrix dotadd(imatrix result, imatrix op1, imatrix op2) {
        return GPU.KERNEL_MATRIX_OP.dotadd(result, op1, op2);
    }

    /**
     * Calculates the element by element addition of factor1 * op1 and factor2 *
     * op2.
     *
     * @param result the matrix to store the result.
     * @param factor1 the first factor.
     * @param op1 the first operand.
     * @param factor2 the second factor.
     * @param op2 the second operand.
     * @return the result matrix
     */
    @Override
    public imatrix dotadd(imatrix result, float factor1, imatrix op1, float factor2, imatrix op2) {
        return GPU.KERNEL_MATRIX_OP.dotadd(result, factor1, op1, factor2, op2);
    }

    /**
     * Calculates the element by element subtraction of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    @Override
    public imatrix dotsubtract(imatrix result, imatrix op1, imatrix op2) {
        return GPU.KERNEL_MATRIX_OP.dotsubtract(result, op1, op2);
    }

    /**
     * Calculates the element by element subtraction of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    @Override
    public imatrix dotmultiply(imatrix result, imatrix op1, imatrix op2) {
        return GPU.KERNEL_MATRIX_OP.dotmultiply(result, op1, op2);
    }

    public static void cleanup() {
        clReleaseCommandQueue(GPU.CL_COMMAND_QUEUE);
        clReleaseContext(GPU.CL_CONTEXT);
    }

    public static cl_mem createReadMem(imatrix matrix, int padcol, int padrow) {
        cl_mem mem = clCreateBuffer(GPU.CL_CONTEXT, CL_MEM_READ_ONLY,
                (matrix.getNrOfRows() + padrow) * (matrix.getNrOfColumns() + padcol) * matrix.getNrOfSlices()
                * Sizeof.cl_float, null, null);
        return mem;
    }

    public static cl_mem createReadWriteMem(imatrix matrix, int padcol, int padrow) {
        cl_mem mem = clCreateBuffer(GPU.CL_CONTEXT, CL_MEM_READ_WRITE,
                (matrix.getNrOfRows() + padrow) * (matrix.getNrOfColumns() + padcol) * matrix.getNrOfSlices()
                * Sizeof.cl_float, null, null);
        return mem;
    }
}
