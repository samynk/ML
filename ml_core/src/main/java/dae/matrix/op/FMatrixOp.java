/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.op;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import dae.neuralnet.activation.ActivationFunction;

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
     * Performs the back propagation correlation operation in batch.
     *
     * @param input The input matrix.
     * @param filter The filter matrix.
     * @param stride the stride
     * @param output The output matrix.
     */
    public void batchBackpropCorrelate(imatrix input, imatrix filter, int stride, imatrix output);

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
    public void batchMaxPool(imatrix input, imatrix output, intmatrix maskLayer);

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
    public void batchBackpropMaxPool(imatrix input, intmatrix maskLayer, int scaleX, int scaleY, imatrix output);

    /**
     * Calculates a fuzzification layer.
     *
     * @param input the inputs to fuzzify.
     * @param classes then number of classes.
     * @param a the weights that determine the slopes of the transition.
     * @param b the weights that determine the crossing point between two
     * classes.
     * @param output the fuzzified input.
     */
    public void fuzzyFunction(imatrix input, int classes, imatrix a, imatrix b, imatrix output);

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
     * @param classes the number of fuzzy classes.
     * @param output the output matrix which is also a row vector.
     */
    public void fuzzyShiftMinus(imatrix input, int classes, imatrix output);

    /**
     * Converts a one D vector into a 2D matrix with the following formula:
     *
     * nrOfVariables : input.rows / classes.
     *
     * The output is the a 2d matrix with dimensions : [nrOfVariables,classes-1]
     *
     * Per row of the output matrix the follow formula will be applied:
     * output[currentRow] = [input1 - input0, input2 - input1, ...,
     * input_classes-1 - input_classes_2]
     *
     * resulting in a row with (classes-1) elements.
     *
     * @param input the input matrix.
     * @param classes the number of classes.
     * @param output the 2D output matrix.
     */
    public void fuzzyShiftDeltas(imatrix input, int classes, imatrix output);

    /**
     * Performs a back propagation into the deltas of the previous layer.
     *
     * @param input the deltas of the fuzzification layer, in batch.
     * @param weights normally, the a-weights of the fuzzification layer.
     * @param classes the number of classes in the fuzzification layer.
     * @param output the output, in batch.
     */
    public void fuzzyBackProp(imatrix input, imatrix weights, int classes, imatrix output);

    public void fuzzyInputAdd(imatrix inputs, imatrix weights, int classes, imatrix deltas);

    /**
     * Calculates the gradients for the a
     */
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

    /**
     * Calculates the element by element addition of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    public imatrix dotadd(imatrix result, imatrix op1, imatrix op2);

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
    public imatrix dotadd(imatrix result, float factor1, imatrix op1, float factor2, imatrix op2);

    /**
     * Calculates the sum per row and per hyperslice and stores the sum into the
     * corresponding row of the output matrix.
     *
     * @param input the input matrix.
     * @param output the output matrix.
     */
    public void sumPerRow(imatrix input, imatrix output);

    /**
     * Calculates the element by element subtraction of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    public imatrix dotsubtract(imatrix result, imatrix op1, imatrix op2);

    /**
     * Calculates the element by element multiplication of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    public imatrix dotmultiply(imatrix result, imatrix op1, imatrix op2);

    /**
     * Multiplies every element of the op1 matrix with the given factor.
     *
     * @param result the matrix to store the result.
     * @param op1 the matrix to multiply with the factor.
     * @param factor the factor to multipy the matrix with.
     * @return the result matrix
     */
    public imatrix dotmultiply(imatrix result, imatrix op1, float factor);

    /**
     * Squares the matrix.
     *
     * @param op1 the matrix to square.
     * @return the squared matrix.
     */
    public imatrix square(imatrix op1);

    /**
     * Copies one matrix into another matrix. The number of rows,columns slices
     * and hyperslices copied is the minimum of the corresponding dimensions of
     * both matrices.
     *
     * @param toCopy the matrix to copy.
     * @param dest the destination matrix.
     */
    public void copyInto(imatrix toCopy, imatrix dest);

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
    public void copyIntoSlice(imatrix toCopy, imatrix dest);

    /**
     * Applies the activation function on the given matrix.
     *
     * @param function the function that defines the derived activation
     * function.
     * @param m the matrix to apply the activation function to.
     */
    public void applyActivation(ActivationFunction function, fmatrix m);

    /**
     * Applies the derived activation function on the give matrix.
     *
     * @param function the function that defines the derived activation
     * function.
     * @param m the matrix to apply the activation function to.
     */
    public void applyDerivedActivation(ActivationFunction function, fmatrix m);

    /**
     * Resets a matrix to zero.
     *
     * @param m the matrix to reset.
     */
    public void reset(fmatrix m);

    /**
     * Randomizes a matrix between the given bound.
     *
     * @param m the matrix to randomize.
     * @param min the minimum for the random float.
     * @param max the maximum for the random float.
     */
    public void randomize(imatrix m, float min, float max);

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
     * @return the resulting updated matrix.
     */
    public imatrix adamVelocity(imatrix result, float beta2, imatrix previousVelocity, imatrix gradient);

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
    public imatrix adamAdaptWeights(imatrix weights, float eta, float beta1, float beta2, float epsilon, imatrix moment, imatrix velocity);

}
