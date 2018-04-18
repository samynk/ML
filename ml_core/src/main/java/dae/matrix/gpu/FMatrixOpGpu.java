package dae.matrix.gpu;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import dae.matrix.mdim2D;
import dae.matrix.op.FMatrixOp;
import dae.neuralnet.activation.ActivationFunction;
import static org.jocl.CL.clCreateBuffer;
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

    private final mdim2D aDim = new mdim2D();
    private final mdim2D bDim = new mdim2D();
    private final mdim2D cDim = new mdim2D();

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
        determineLayout(A, aDim);
        determineLayout(B, bDim);
        determineLayout(C, cDim);

        int M = cDim.rows;
        int K = bDim.rows;
        int N = cDim.columns;

        FloatDeviceBuffer ADB = A.getDeviceBuffer();
        FloatDeviceBuffer BDB = B.getDeviceBuffer();
        FloatDeviceBuffer CDB = C.getDeviceBuffer();
        cl_mem memA = ADB.upload();
        cl_mem memB = BDB.upload();
        cl_mem memC = CDB.upload();
        // Execute GEMM:
        // C = alpha * A * B + beta * C
        cl_event event = new cl_event();
        CLBlastSgemm(CLBlastLayoutColMajor,
                aDim.transposed ? CLBlastTransposeYes : CLBlastTransposeNo,
                bDim.transposed ? CLBlastTransposeYes : CLBlastTransposeNo,
                M, N, K,
                alpha,
                memA, 0, aDim.ld,
                memB, 0, bDim.ld,
                beta,
                memC, 0, cDim.ld,
                GPU.CL_COMMAND_QUEUE, event);

        CDB.markGpuAsMaster();
    }

    private void determineLayout(imatrix m, mdim2D dim) {
        if (m.isBatchMatrix()) {
            dim.transposed = m.isTransposed();
            if (m.isTransposed()) {
                dim.rows = m.getNrOfHyperSlices();
                dim.columns = m.getNrOfColumns();
                dim.ld = dim.columns;
            } else {
                dim.rows = m.getNrOfRows();
                dim.columns = m.getNrOfHyperSlices();
                dim.ld = dim.rows;
            }
        } else {
            dim.rows = m.getNrOfRows();
            dim.columns = m.getNrOfColumns();
            dim.transposed = m.isTransposed();
            if (m.isTransposed()) {
                dim.ld = dim.columns;
            } else {
                dim.ld = dim.rows;
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
     * Calculates the relu activation function. The result is stored back into
     * the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void relu(imatrix O) {
        GPU.KERNEL_ACTIVATION.relu(O);
    }

    /**
     * Calculates the derivative of the relu activation function. The result is
     * stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void drelu(imatrix O) {
        GPU.KERNEL_ACTIVATION.drelu(O);
    }

    /**
     * Calculates the leaky relu activation function. The result is stored back
     * into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void leakyrelu(imatrix O) {
        GPU.KERNEL_ACTIVATION.leakyrelu(O);
    }

    /**
     * Calculates the derivative of the leaky relu activation function. The
     * result is stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void dleakyrelu(imatrix O) {
        GPU.KERNEL_ACTIVATION.dleakyrelu(O);
    }

    /**
     * Calculates the tanh activation function. The result is stored back into
     * the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void tanh(imatrix O) {
        GPU.KERNEL_ACTIVATION.tanh(O);
    }

    /**
     * Calculates the derivative of the relu activation function. The result is
     * stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void dtanh(imatrix O) {
        GPU.KERNEL_ACTIVATION.dtanh(O);
    }

    /**
     * Calculates the derivative of the identity activation function. The result
     * is stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    public void didentity(imatrix O) {
        GPU.KERNEL_ACTIVATION.didentity(O);
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
     * @param scaleX the x-scale of the pool layer.
     * @param scaleY the y-scale of the pool layer.
     * @param output the output matrix.
     *
     */
    @Override
    public void batchBackpropMaxPool(imatrix input, intmatrix maskLayer, int scaleX, int scaleY, imatrix output) {
        GPU.KERNEL_POOL.backpropMaxPool(input, maskLayer, scaleX, scaleY, output);
    }
    
    /**
     * The input contains data per two slices. The even slices contain the value 
     * data, the oneven slices contain the rotation data. The max pooling is 
     * applied to the value data, with conservation of the rotation data.
     * @param input the input matrix with value and rotation slices.
     * @param output the scale output matrix with value and rotation slices.
     * @param maskLayer the layer that contains the cells with the maximum value.
     */
    @Override
    public void maxRotationPool(imatrix input, imatrix output, intmatrix maskLayer){
        GPU.KERNEL_POOL.maxRotationPool(input, output, maskLayer);
    }
    
    /**
     * Transfers the maximum values into the output matrix at the correct cell
     * location as indicated by the masklayer.
     *
     * @param input the input matrix, a downscaled version of the output matrix.
     * @param maskLayer the mask layer that guides the transfer of values to the
     * output matrix.
     * @param output the output matrix.
     */
    @Override
    public void backpropMaxRotationPool(imatrix input, intmatrix maskLayer, imatrix output){
         GPU.KERNEL_POOL.backpropMaxRotationPool(input, maskLayer, output);
    }

    @Override
    public void fuzzyFunction(imatrix input, int classes, imatrix a, imatrix b, imatrix output) {
        GPU.KERNEL_FUZZY.fuzzyFunction(input, classes, a, b, output);
    }

    /**
     * Performs a back propagation into the deltas of the previous layer.
     *
     * @param input the deltas of the fuzzification layer, in batch.
     * @param weights normally, the a-weights of the fuzzification layer.
     * @param classes the number of classes in the fuzzification layer.
     * @param output the output, in batch.
     */
    @Override
    public void fuzzyBackProp(imatrix input, imatrix weights, int classes, imatrix output) {
        GPU.KERNEL_FUZZY.fuzzyBackProp(input, weights, classes, output);
    }

    /**
     * Expands the elements of the input into the output with the following
     * algorithm:
     *
     * o1 = 1-i1 o2 = i1-i2 o3 = i3-i2 ... on = i(n-1)
     *
     * This also means that for every classes-1 input elements an extra output
     * element will be created.
     *
     * size(outputs) = classes * (size(inputs)/(classes-1))
     *
     * @param input the input matrix which is a row vector.
     * @param output the output matrix which is also a row vector.
     */
    @Override
    public void fuzzyShiftMinus(imatrix input, int classes, imatrix output) {
        GPU.KERNEL_FUZZY.fuzzyShiftMinus(input, classes, output);
    }

    @Override
    public void fuzzyInputAdd(imatrix inputs, imatrix weights, int classes, imatrix deltas) {
        GPU.KERNEL_FUZZY.fuzzyInputAdd(inputs, weights, classes, deltas);
    }

    /**
     * Converts a one D vector into a 2D matrix with the following formula:
     *
     * nrOfVariables : input.rows / classes.
     *
     * The output is the a one D matrix with dimensions :
     * [nrOfVariables*(classes-1)]
     *
     * Per row of the output matrix the follow formula will be applied:
     * output[0] = input1 - input0 output[1] = input2 - input1, ..., ...
     * output[input_classes-1 - input_classes_2]
     *
     * resulting in a row with (classes-1) elements.
     *
     * @param input the input matrix.
     * @param classes the number of classes.
     * @param output the 2D output matrix.
     */
    @Override
    public void fuzzyShiftDeltas(imatrix input, int classes, imatrix output) {
        GPU.KERNEL_FUZZY.fuzzyShiftDeltas(input, classes, output);
    }

    /**
     * Calculates the element by element addition of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    @Override
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
     * Calculates the sum per row and per hyperslice and stores the sum into the
     * corresponding row of the output matrix.
     *
     * @param input the input matrix.
     * @param output the output matrix.
     */
    @Override
    public void sumPerRow(imatrix input, imatrix output) {
        GPU.KERNEL_MATRIX_OP.sumPerRow(input, output);
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

    /**
     * Multiplies every element of the op1 matrix with the given factor.
     *
     * @param result the matrix to store the result.
     * @param op1 the matrix to multiply with the factor.
     * @param factor the factor to multipy the matrix with.
     * @return the result matrix
     */
    @Override
    public imatrix dotmultiply(imatrix result, imatrix op1, float factor) {
        return GPU.KERNEL_MATRIX_OP.dotmultiply(result, op1, factor);
    }

    /**
     * Squares the matrix.
     *
     * @param op1 the matrix to square.
     * @return the squared matrix.
     */
    @Override
    public imatrix square(imatrix op1) {
        return GPU.KERNEL_MATRIX_OP.square(op1);
    }

    /**
     * Calculates the new velocity in the adam algorithm by applying the
     * following formula to every cell in the matrix:
     *
     * newVelocity = beta2*previousVelocity + (1-beta2) * gradient^2
     *
     * @param result the imatrix to store the result in.
     * @param beta2 the beta2 factor of the algorithm.
     * @param previousVelocity the previous velocity matrix.
     * @param gradient the current gradient.
     */
    @Override
    public imatrix adamVelocity(imatrix result, float beta2, imatrix previousVelocity, imatrix gradient) {
        return GPU.KERNEL_MATRIX_OP.adamVelocity(result, beta2, previousVelocity, gradient);
    }

    /**
     * Adapts the weights according to the adam gradient descent algorithm. The
     * bias correction will be applied in place.
     *
     * @param weights the current weights.
     * @param eta the learning rate.
     * @param beta1 the beta1 value.
     * @param beta2 the beta2 value.
     * @param epsilon the epsilon value.
     * @param moment the current moment.
     * @param velocity the current velocity.
     * @return
     */
    @Override
    public imatrix adamAdaptWeights(imatrix weights, float eta, float beta1, float beta2, float epsilon, imatrix moment, imatrix velocity) {
        return GPU.KERNEL_MATRIX_OP.adamAdaptWeights(weights, eta, beta1, beta2, epsilon, moment, velocity);
    }

    /**
     * Copies one matrix into another matrix. The number of rows,columns slices
     * and hyperslices copied is the minimum of the corresponding dimensions of
     * both matrices.
     *
     * @param toCopy the matrix to copy.
     * @param dest the destination matrix.
     */
    @Override
    public void copyInto(imatrix toCopy, imatrix dest) {
        GPU.enqueueCopyMatrix(toCopy, dest);
    }

    /**
     * Copies a matrix into another matrix. The slice sizes are compared and if
     * the slice size is the same, the slices will be copied regardless of the
     * rows and columns of the two matrices.
     *
     * This function can be used to copy a column vector into a 2D matrix for
     * example.
     *
     * @param toCopy the matrix to copy.
     * @param dest the destination matrix.
     */
    @Override
    public void copyIntoSlice(imatrix toCopy, imatrix dest) {
        GPU.enqueueCopySliceMatrix(toCopy, dest);
    }

    /**
     * Copies the slices of matrix1 and matrix2 into the destination matrix. One
     * slice of the destination matrix will be composed of the concatenation of
     * a slice of the first matrix and a slice of the second matrix.
     *
     * @param matrix1 the first matrix.
     * @param matrix2 the second matrix.
     * @param dest the destination matrix.
     */
    @Override
    public void zip(imatrix matrix1, imatrix matrix2, imatrix dest) {
        GPU.zip(matrix1, matrix2, dest);
    }

    /**
     * Unzips the src matrix into two destination matrices per slice. The even
     * slices will be copied into the first destination matrix, the uneven
     * slices will be copied into the second destination matrix.
     *
     * @param src the source matrix.
     * @param dest1 the first destination matrix.
     * @param dest2 the second destination matrix.
     */
    @Override
    public void unzip(imatrix src, imatrix dest1, imatrix dest2) {
        GPU.unzip(src, dest1, dest2);
    }

    /**
     * Applies the activation function on the given matrix.
     *
     * @param function the function that defines the derived activation
     * function.
     * @param m the matrix to apply the activation function to.
     */
    @Override
    public void applyActivation(ActivationFunction function, fmatrix m) {
        switch (function) {
            case SIGMOID:
            case CESIGMOID:
                sigmoid(m);
                break;
            case RELU:
                relu(m);
                break;
            case LEAKYRELU:
                leakyrelu(m);
                break;
            case TANH:
                tanh(m);
                break;
        }
    }

    /**
     * Applies the derived activation function on the give matrix.
     *
     * @param function the function that defines the derived activation
     * function.
     * @param m the matrix to apply the activation function to.
     */
    @Override
    public void applyDerivedActivation(ActivationFunction function, fmatrix m) {
        switch (function) {
            case SIGMOID:
                dsigmoid(m);
                break;
            case CESIGMOID:
            case IDENTITY:
            case SOFTMAX:
                didentity(m);
                break;
            case RELU:
                drelu(m);
                break;
            case LEAKYRELU:
                dleakyrelu(m);
                break;
            case TANH:
                dtanh(m);
                break;
        }
    }

    /**
     * Rotates a kernel. The first slice will be preserved and rotated copies
     * will be generated in the subsequent slices. The start angle indicates the
     * angle of the first slice.
     *
     * @param filter the kernel to rotate.
     * @param nrOfRotations the number of rotations.
     * @param minAngle the start angle, first slice included.
     * @param maxAngle the end angle.
     * @param output the output of the rotation.
     */
    @Override
    public void rotateKernels(imatrix filter, int nrOfRotations, float minAngle, float maxAngle, imatrix output) {
        GPU.KERNEL_CONVOLV.rotateKernels(filter, nrOfRotations, minAngle, maxAngle, output);
    }

    /**
     * Accumulates the output rotations into a kernel. The start angle indicates
     * the angle of the first slice.
     *
     * @param rotatedOutput the kernel to rotate.
     * @param nrOfRotations the number of rotations.
     * @param minAngle the start angle, first slice included.
     * @param maxAngle the end angle.
     * @param kernelOutput the output of the rotation.
     */
    @Override
    public void accumulateRotateKernels(imatrix rotatedOutput, int nrOfRotations, float minAngle, float maxAngle, imatrix kernelOutput) {
        GPU.KERNEL_CONVOLV.accumulateRotateKernels(rotatedOutput, nrOfRotations, minAngle, maxAngle, kernelOutput);
    }

    /**
     * Condenses the input to detect the max activation rotation. The maximum
     * rotation and activation value is then stored into the output matrix.
     *
     * @param input the input matrix.
     * @param nrOfFeatures number of features in the convolution.
     * @param nrOfRotations number of rotations per feature.
     * @param minAngle the minimum angle of the rotation.
     * @param maxAngle the maximum angle of the rotation.
     * @param valOutput the output matrix of this function that contains the
     * maximum activation values.
     * @param rotOutput the output matrix of this function that contains the
     * rotation values.
     */
    @Override
    public void maxRotation(imatrix input, int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix valOutput, imatrix rotOutput) {
        GPU.KERNEL_CONVOLV.maxRotation(input, nrOfFeatures, nrOfRotations, minAngle, maxAngle, valOutput, rotOutput);
    }

    /**
     * Performs the inverse operation of the maxRotation and stores the given
     * value according the the rotation stored in rotInput
     *
     * @param valInput the activation values.
     * @param rotInput the rotation values.
     * @param nrOfFeatures the number of features in the convolution layer.
     * @param nrOfRotations the number of rotations per features.
     * @param minAngle the minAngle of the rotations.
     * @param maxAngle the maxAngle of the rotations.
     * @param output the result of the inverse operation.
     */
    @Override
    public void maxInverseRotation(imatrix valInput, imatrix rotInput, int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix output) {
        GPU.KERNEL_CONVOLV.maxInverseRotation(valInput, rotInput, nrOfFeatures, nrOfRotations, minAngle, maxAngle, output);
    }

    @Override
    public void reset(fmatrix m) {
        GPU.zeroFill(m);
    }

    /**
     * Randomizes a matrix between the given bound.
     *
     * @param m the matrix to randomize.
     * @param min the minimum for the random float.
     * @param max the maximum for the random float.
     */
    @Override
    public void randomize(imatrix m, float min, float max) {
        GPU.KERNEL_MATRIX_OP.randomize(m, min, max);
    }

    public static void cleanup() {
        clReleaseCommandQueue(GPU.CL_COMMAND_QUEUE);
        clReleaseContext(GPU.CL_CONTEXT);
    }

    public static cl_mem createMem(imatrix cpuBuffer, int padding, long mode) {
        int zp = cpuBuffer.getZeroPadding();
        int totalSize = (cpuBuffer.getNrOfRows() + 2 * zp)
                * (cpuBuffer.getNrOfColumns() + 2 * zp)
                * cpuBuffer.getNrOfSlices()
                * cpuBuffer.getNrOfHyperSlices() + padding;
        cl_mem mem = clCreateBuffer(GPU.CL_CONTEXT, mode,
                totalSize * Sizeof.cl_float, null, null);
        return mem;
    }
}
