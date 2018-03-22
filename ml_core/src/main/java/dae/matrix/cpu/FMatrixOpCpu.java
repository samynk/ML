/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.cpu;

import dae.matrix.fmatrix;
import static dae.matrix.fmatrix.equalDimension;
import dae.matrix.gpu.GPU;
import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import dae.matrix.op.FMatrixOp;
import dae.neuralnet.activation.ActivationFunction;
import java.nio.FloatBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FMatrixOpCpu implements FMatrixOp {

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
    @Override
    public void convolve(imatrix input, imatrix filter, int stride, imatrix output) {
        int zeroPadding = input.getZeroPadding();

        for (int inputSlice = 0; inputSlice < input.getNrOfSlices(); ++inputSlice) {
            for (int or = 0; or < output.getNrOfRows(); ++or) {
                for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                    float c = convolveSingle(or, oc, 0, input, inputSlice, zeroPadding, filter, stride);
                    // one filter, so outputslice == inputslice
                    output.set(or, oc, inputSlice, c);
                }
            }
        }
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
        int zeroPadding = input.getZeroPadding();
        int filtersPerInputSlice = filter.getNrOfSlices() / input.getNrOfSlices();

        for (int filterSlice = 0; filterSlice < filter.getNrOfSlices(); ++filterSlice) {
            int inputSlice = filterSlice / filtersPerInputSlice;
            for (int or = 0; or < output.getNrOfRows(); ++or) {
                for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                    float c = convolveSingle(or, oc, filterSlice, input, inputSlice, zeroPadding, filter, stride);
                    output.set(or, oc, filterSlice, c);
                }
            }
        }
    }

    /**
     * Applies a max pool on the input matrix and stores it into the output
     * matrix. It is assumed that the dimensions of the output matrix are
     * dividers of the dimensions of the input matrix.
     *
     * @param input the input matrix.
     * @param output the output matrix.
     */
    @Override
    public void batchMaxPool(imatrix input, imatrix output, intmatrix maskLayer) {
        // set all values to zero
        maskLayer.reset();

        int scaleX = input.getNrOfColumns() / output.getNrOfColumns();
        int scaleY = input.getNrOfRows() / output.getNrOfRows();

        int slices = Math.min(input.getNrOfSlices(), output.getNrOfSlices());

        for (int slice = 0; slice < slices; ++slice) {
            for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                for (int or = 0; or < output.getNrOfRows(); ++or) {
                    float m = max(slice, or, oc, scaleX, scaleY, input, maskLayer);
                    output.set(or, oc, slice, m);
                }
            }
        }
    }

    private float max(int slice, int or, int oc, int filterCols, int filterRows, imatrix input, intmatrix maskLayer) {
        float m = -Float.MAX_VALUE;
        int imx = 0;
        int imy = 0;
        int cell = 0;
        for (int c = 0; c < filterCols; ++c) {
            for (int r = 0; r < filterRows; ++r) {
                int ix = or * filterRows + r;
                int iy = oc * filterRows + c;
                float value = input.get(ix, iy, slice);
                if (value > m) {
                    m = value;
                    cell = r + c * filterRows;
                }

            }
        }
        maskLayer.set(or, oc, slice, cell);
        return m;
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
        int zeroPadding = input.getZeroPadding();
        int filtersPerInputSlice = filter.getNrOfSlices() / input.getNrOfSlices();

        for (int filterSlice = 0; filterSlice < filter.getNrOfSlices(); ++filterSlice) {
            int inputSlice = filterSlice / filtersPerInputSlice;
            for (int or = 0; or < output.getNrOfRows(); ++or) {
                for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                    float c = correlateSingle(or, oc, filterSlice, input, inputSlice, zeroPadding, filter, stride);
                    output.set(or, oc, filterSlice, c);
                }
            }
        }
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
        int slicesPerOutput = filter.getNrOfSlices() / output.getNrOfSlices();
        int zeroPadding = input.getZeroPadding();
        for (int oSlice = 0; oSlice < output.getNrOfSlices(); ++oSlice) {
            for (int inputSlice = 0; inputSlice < slicesPerOutput; ++inputSlice) {
                int currentSlice = oSlice * slicesPerOutput + inputSlice;
                for (int or = 0; or < output.getNrOfRows(); ++or) {
                    for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                        float c = correlateSingle(or, oc, currentSlice, input, currentSlice, zeroPadding, filter, stride);
                        float current = output.get(or, oc, oSlice);
                        output.set(or, oc, oSlice, current + c);
                    }
                }
            }
        }
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
        int slices = Math.min(input.getNrOfSlices(), output.getNrOfSlices());

        for (int slice = 0; slice < slices; ++slice) {
            for (int ic = 0; ic < input.getNrOfColumns(); ++ic) {
                for (int ir = 0; ir < input.getNrOfRows(); ++ir) {
                    int oc = ic * scaleX;
                    int or = ir * scaleY;
                    int cell = maskLayer.get(ir, ic, slice);
                    int x = cell / scaleY;
                    int y = cell % scaleY;
                    float v = input.get(ir, ic, slice);
                    output.set(or + y, oc + x, slice, v);
                }
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
    private float convolveSingle(int or, int oc, int filterSlice, imatrix input, int inputSlice, int zeroPadding, imatrix filter, int stride) {
        int irb = or * stride;
        int irc = oc * stride;

        float c = 0;
        for (int fr = 0; fr < filter.getNrOfRows(); ++fr) {
            for (int fc = 0; fc < filter.getNrOfColumns(); ++fc) {
                int ir = irb + fr - zeroPadding;
                int ic = irc + fc - zeroPadding;
                if (ir >= 0 && ic >= 0 && ir < input.getNrOfRows() && ic < input.getNrOfColumns()) {
                    float iv = input.get(ir, ic, inputSlice);
                    float fv = filter.get(fr, fc, filterSlice);
                    c += iv * fv;
                }
            }
        }
        return c;
    }

    /**
     * Performs the correlation for a single cell in the output.
     *
     * @param or the current row of the output.
     * @param oc the current column of the output.
     * @param input the input matrix.
     * @param filter the filter.
     * @param stride the stride of the convolution window.
     * @return
     */
    private float correlateSingle(int or, int oc, int slice, imatrix input, int inputSlice, int zeroPadding, imatrix filter, int stride) {
        int irb = or * stride;
        int irc = oc * stride;
        int fcs = filter.getNrOfColumns();
        int frs = filter.getNrOfRows();
        float sum = 0;
        for (int fr = 0; fr < filter.getNrOfRows(); ++fr) {
            int ir = irb + fr - zeroPadding;
            for (int fc = 0; fc < filter.getNrOfColumns(); ++fc) {
                int ic = irc + fc - zeroPadding;
                if (ir >= 0 && ic >= 0 && ir < input.getNrOfRows() && ic < input.getNrOfColumns()) {
                    float iv = input.get(ir, ic, inputSlice);
                    float fv = filter.get(frs - fr - 1, fcs - fc - 1, slice);
                    sum += iv * fv;
                }
            }
        }
        return sum;
    }

    /**
     * Calculates a fuzzification layer.
     *
     * @param input the inputs to fuzzify.
     * @param a the weights that determine the slopes of the transition.
     * @param b the weights that determine the crossing point between two
     * classes.
     * @param functions the fuzzified input.
     */
    @Override
    public void fuzzyFunction(imatrix input, int classes, imatrix a, imatrix b, imatrix functions) {
        int actualClasses = classes - 1;
        for (int h = 0; h < input.getNrOfHyperSlices(); ++h) {
            for (int ir = 0; ir < input.getNrOfRows(); ++ir) {
                float iv = input.get(ir, 0, 0, h);
                for (int oc = 0; oc < actualClasses; ++oc) {
                    int index = ir * actualClasses + oc;
                    float av = a.get(index, 0);
                    float bv = b.get(index, 0);

                    float v = 1 / (1 + (float) Math.exp(-av * (iv + bv)));
                    functions.set(index, 0, 0, h, v);
                }
            }
        }
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
        int numVars = input.getNrOfRows() / (classes - 1);
        int hyperSlices = Math.min(input.getNrOfHyperSlices(), output.getNrOfHyperSlices());
        for (int h = 0; h < hyperSlices; ++h) {
            int oRow = 0;
            for (int rv = 0; rv < numVars; ++rv) {
                float previous = 1;
                int iBase = rv * (classes - 1);
                for (int ic = 0; ic < (classes - 1); ++ic) {
                    float iv = input.get(iBase + ic, 0, 0, h);
                    output.set(oRow++, 0, 0, h, previous - iv);
                    previous = iv;
                }
                output.set(oRow++, 0, 0, h, previous);
            }
        }
    }

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
    @Override
    public void fuzzyShiftDeltas(imatrix input, int classes, imatrix output) {
        int nrOfVariables = input.getNrOfRows() / classes;
        for (int h = 0; h < input.getNrOfHyperSlices(); ++h) {
            for (int v = 0; v < nrOfVariables; ++v) {
                int iBase = v * classes;
                int oBase = v * (classes - 1);
                for (int c = 0; c < classes - 1; ++c) {
                    float dn = input.get(iBase + c, 0, 0, h);
                    float dnp1 = input.get(iBase + c + 1, 0, 0, h);

                    output.set(oBase + c, 0, 0, h, dnp1 - dn);
                }
            }
        }
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
        for (int h = 0; h < input.getNrOfHyperSlices(); ++h) {
            for (int r = 0; r < output.getNrOfRows(); ++r) {
                float sum = 0;
                int inputRow = r * (classes - 1);
                for (int c = 0; c < classes - 1; ++c) {
                    float w = weights.get(inputRow + c, 0, 0, 0);
                    float i = input.get(inputRow + c, 0, 0, h);
                    sum += w * i;
                }
                output.set(r, 0, 0, h, sum);
            }
        }
    }

    @Override
    public void fuzzyInputAdd(imatrix inputs, imatrix weights, int classes, imatrix deltas) {
        for (int h = 0; h < inputs.getNrOfHyperSlices(); ++h) {
            for (int r = 0; r < inputs.getNrOfRows(); ++r) {
                float iv = inputs.get(r, 0, 0, h);
                for (int c = 0; c < (classes - 1); ++c) {
                    int wdIndex = r * (classes - 1) + c;
                    float current = weights.get(wdIndex, 0);
                    deltas.set(wdIndex, 0, 0, h, current + iv);
                }
            }
        }
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
        m.applyFunction(function.getActivation());
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
        m.applyFunction(function.getDerivedActivation());
    }

    /**
     * Calculates the sigmoid activation function. The result is stored back
     * into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    @Override
    public void sigmoid(imatrix O) {
        O.applyFunction(x -> 1 / (1 + (float) Math.exp(-x)));
    }

    /**
     * Calculates the derivative of the sigmoid activation function. The result
     * is stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    @Override
    public void dsigmoid(imatrix O) {
        O.applyFunction(x -> 1 / (1 + (float) Math.exp(-x)));
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
        for (int h = 0; h < result.getNrOfHyperSlices(); ++h) {
            for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
                for (int row = 0; row < result.getNrOfRows(); ++row) {
                    for (int column = 0; column < result.getNrOfColumns(); ++column) {
                        float op1value = op1.get(row, column, slice, h);
                        float op2value = op2.get(row, column, slice, h);
                        result.set(row, column, slice, op1value + op2value);
                    }
                }
            }
        }
        return result;
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
        for (int h = 0; h < result.getNrOfHyperSlices(); ++h) {
            for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
                for (int row = 0; row < result.getNrOfRows(); ++row) {
                    for (int column = 0; column < result.getNrOfColumns(); ++column) {
                        float op1value = op1.get(row, column, slice, h);
                        float op2value = op2.get(row, column, slice, h);
                        result.set(row, column, slice, factor1 * op1value + factor2 * op2value);
                    }
                }
            }
        }
        return result;
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
        for (int r = 0; r < input.getNrOfRows(); ++r) {
            float sum = 0f;
            for (int h = 0; h < input.getNrOfHyperSlices(); ++h) {
                sum += input.get(r, 0, 0, h);
            }
            output.set(r, 0, sum);
        }
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
        for (int h = 0; h < result.getNrOfHyperSlices(); ++h) {
            for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
                for (int row = 0; row < result.getNrOfRows(); ++row) {
                    for (int column = 0; column < result.getNrOfColumns(); ++column) {
                        float op1value = op1.get(row, column, slice, h);
                        float op2value = op2.get(row, column, slice, h);
                        result.set(row, column, slice, op1value - op2value);
                    }
                }
            }
        }
        return result;
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
        for (int hyperslice = 0; hyperslice < result.getNrOfHyperSlices(); ++hyperslice) {
            for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
                for (int row = 0; row < result.getNrOfRows(); ++row) {
                    for (int column = 0; column < result.getNrOfColumns(); ++column) {
                        float op1value = op1.get(row, column, slice, hyperslice);
                        float op2value = op2.get(row, column, slice, hyperslice);
                        result.set(row, column, slice, hyperslice, op1value * op2value);
                    }
                }
            }
        }

        return result;
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
        for (int hyperslice = 0; hyperslice < result.getNrOfHyperSlices(); ++hyperslice) {
            for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
                for (int row = 0; row < result.getNrOfRows(); ++row) {
                    for (int column = 0; column < result.getNrOfColumns(); ++column) {
                        float op1value = op1.get(row, column, slice, hyperslice);
                        result.set(row, column, slice, hyperslice, op1value * factor);
                    }
                }
            }
        }
        return result;
    }

    /**
     * Squares the matrix.
     *
     * @param op1 the matrix to square.
     * @return the squared matrix (for chaining purposes).
     */
    @Override
    public imatrix square(imatrix op1) {
        op1.applyFunction(x -> x * x);
        return op1;
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
     * @return the resulting updated matrix.
     */
    @Override
    public imatrix adamVelocity(imatrix result, float beta2, imatrix previousVelocity, imatrix gradient) {
        for (int hyperslice = 0; hyperslice < result.getNrOfHyperSlices(); ++hyperslice) {
            for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
                for (int row = 0; row < result.getNrOfRows(); ++row) {
                    for (int column = 0; column < result.getNrOfColumns(); ++column) {
                        float v = previousVelocity.get(row, column, slice, hyperslice);
                        float g = gradient.get(row, column, slice, hyperslice);
                        result.set(row, column, slice, hyperslice, beta2 * v + (1 - beta2) * g * g);
                    }
                }
            }
        }

        return result;
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
        float invOneMinusBeta1 = 1 / (1 - beta1);
        float invOneMinusBeta2 = 1 / (1 - beta2);

        for (int hyperslice = 0; hyperslice < weights.getNrOfHyperSlices(); ++hyperslice) {
            for (int slice = 0; slice < weights.getNrOfSlices(); ++slice) {
                for (int row = 0; row < weights.getNrOfRows(); ++row) {
                    for (int column = 0; column < weights.getNrOfColumns(); ++column) {
                        float v = velocity.get(row, column, slice, hyperslice);
                        float m = moment.get(row, column, slice, hyperslice);
                        float w = weights.get(row, column, slice, hyperslice);
                        float newW = w - (eta * m * invOneMinusBeta1) / ((float) (Math.sqrt(v * invOneMinusBeta2)) + epsilon);
                        weights.set(row, column, slice, hyperslice, newW);
                    }
                }
            }
        }
        return weights;
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

    }

    @Override
    public void copyInto(imatrix toCopy, imatrix dest) {
        if (toCopy.isTransposed() == dest.isTransposed()
                && equalDimension(toCopy, dest)) {
            FloatBuffer srcData = toCopy.getHostData();
            FloatBuffer destData = dest.getHostData();
            srcData.rewind();
            destData.rewind();
            destData.put(srcData);

        } else {
            int eSrcRows = toCopy.getNrOfRows();
            int eDstRows = dest.getNrOfRows();

            int eSrcCols = toCopy.getNrOfColumns();
            int eDstCols = dest.getNrOfColumns();

            int eSrcSlices = toCopy.getNrOfSlices();
            int eDstSlices = dest.getNrOfSlices();

            int eSrcHyperSlices = toCopy.getNrOfHyperSlices();
            int eDstHyperSlices = dest.getNrOfHyperSlices();

            int maxRow = Math.min(eSrcRows, eDstRows);
            int maxCol = Math.min(eSrcCols, eDstCols);
            int maxSlices = Math.min(eSrcSlices, eDstSlices);
            int maxHyperSlices = Math.min(eSrcHyperSlices, eDstHyperSlices);
            for (int hp = 0; hp < maxHyperSlices; ++hp) {
                for (int slice = 0; slice < maxSlices; ++slice) {
                    for (int row = 0; row < maxRow; ++row) {
                        for (int column = 0; column < maxCol; ++column) {
                            float value = toCopy.get(row, column, slice, hp);
                            dest.set(row, column, slice, hp, value);
                        }
                    }
                }
            }
        }
    }

    /**
     * Copies a matrix into another matrix. The slice sizes are compared and if
     * the slice size is the same, the slices will be copied regardless of the
     * rows and columns of the two matrices.
     *
     * This function can be used to copy a column vector into a 2D matrix for
     * example.
     *
     * @param src the matrix to copy.
     * @param dst the destination matrix.
     */
    @Override
    public void copyIntoSlice(imatrix src, imatrix dst) {
        if (src.getSliceSize() == dst.getSliceSize()) {
            int srcSlices = src.getNrOfSlices() * src.getNrOfHyperSlices();
            int dstSlices = dst.getNrOfSlices() * dst.getNrOfHyperSlices();
            int floatsToCopy = Math.min(srcSlices, dstSlices) * src.getSliceSize();
            float[] arrSrc = src.getHostData().array();
            float[] arrDst = dst.getHostData().array();
            System.arraycopy(arrSrc, 0, arrDst, 0, floatsToCopy);
        }else{
            copyInto(src,dst);
        }
    }

    @Override
    public void reset(fmatrix m) {
        m.applyFunction(x -> 0);
    }

}
