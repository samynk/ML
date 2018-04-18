/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.cpu;

import dae.matrix.float2;
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

        for (int h = 0; h < output.getNrOfHyperSlices(); ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                    for (int or = 0; or < output.getNrOfRows(); ++or) {
                        float m = max(or, oc, slice, h, scaleX, scaleY, input, maskLayer);
                        output.set(or, oc, slice, h, m);
                    }
                }
            }
        }
    }

    private float max(int or, int oc, int slice, int h, int filterCols, int filterRows, imatrix input, intmatrix maskLayer) {
        float m = -Float.MAX_VALUE;
        int cell = 0;
        for (int c = 0; c < filterCols; ++c) {
            for (int r = 0; r < filterRows; ++r) {
                int ix = or * filterRows + r;
                int iy = oc * filterRows + c;
                float value = input.get(ix, iy, slice, h);
                if (value > m) {
                    m = value;
                    cell = r + c * filterRows;
                }

            }
        }
        maskLayer.set(or, oc, slice, h, cell);
        return m;
    }

    /**
     * The input contains data per two slices. The even slices contain the value
     * data, the oneven slices contain the rotation data. The max pooling is
     * applied to the value data, with conservation of the rotation data.
     *
     * @param input the input matrix with value and rotation slices.
     * @param output the scale output matrix with value and rotation slices.
     * @param maskLayer the layer that contains the cells with the maximum
     * value.
     */
    @Override
    public void maxRotationPool(imatrix input, imatrix output, intmatrix maskLayer) {
        // set all values to zero
        maskLayer.reset();

        int scaleX = input.getNrOfColumns() / output.getNrOfColumns();
        int scaleY = input.getNrOfRows() / output.getNrOfRows();

        int slices = maskLayer.getNrOfSlices();

        for (int h = 0; h < output.getNrOfHyperSlices(); ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                for (int oc = 0; oc < output.getNrOfColumns(); ++oc) {
                    for (int or = 0; or < output.getNrOfRows(); ++or) {
                        maxRotation(or, oc, slice, h, scaleX, scaleY, input, output, maskLayer);
                    }
                }
            }
        }
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
    public void backpropMaxRotationPool(imatrix input, intmatrix maskLayer, imatrix output) {
        int scaleX = output.getNrOfColumns() / input.getNrOfColumns();
        int scaleY = output.getNrOfRows() / input.getNrOfRows();

        int slices = Math.min(input.getNrOfSlices(), output.getNrOfSlices());
        int hyperSlices = Math.min(input.getNrOfHyperSlices(), output.getNrOfHyperSlices());

        for (int h = 0; h < hyperSlices; ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                for (int ic = 0; ic < input.getNrOfColumns(); ++ic) {
                    for (int ir = 0; ir < input.getNrOfRows(); ++ir) {
                        int oc = ic * scaleX;
                        int or = ir * scaleY;
                        int cell = maskLayer.get(ir, ic, slice / 2, h);
                        int x = cell / scaleY;
                        int y = cell % scaleY;
                        float v = input.get(ir, ic, slice, h);
                        output.set(or + y, oc + x, slice, h, v);
                    }
                }
            }
        }
    }

    private void maxRotation(int or, int oc, int slice, int h, int filterCols, int filterRows, imatrix input, imatrix output, intmatrix maskLayer) {
        float m = -Float.MAX_VALUE;
        float rot = -Float.MAX_VALUE;
        int cell = 0;
        for (int c = 0; c < filterCols; ++c) {
            for (int r = 0; r < filterRows; ++r) {
                int ix = or * filterRows + r;
                int iy = oc * filterRows + c;
                float value = input.get(ix, iy, slice * 2, h);

                if (value > m) {
                    m = value;
                    rot = input.get(ix, iy, slice * 2 + 1, h);
                    cell = r + c * filterRows;
                }

            }
        }
        output.set(or, oc, slice * 2, h, m);
        output.set(or, oc, slice * 2 + 1, h, rot);
        maskLayer.set(or, oc, slice, h, cell);
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
        int hyperSlices = Math.min(input.getNrOfHyperSlices(), output.getNrOfHyperSlices());

        for (int h = 0; h < hyperSlices; ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                for (int ic = 0; ic < input.getNrOfColumns(); ++ic) {
                    for (int ir = 0; ir < input.getNrOfRows(); ++ir) {
                        int oc = ic * scaleX;
                        int or = ir * scaleY;
                        int cell = maskLayer.get(ir, ic, slice, h);
                        int x = cell / scaleY;
                        int y = cell % scaleY;
                        float v = input.get(ir, ic, slice, h);
                        output.set(or + y, oc + x, slice, h, v);
                    }
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

                    float v = av * (iv + bv);
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
     * Rotates a kernel. The start angle indicates the angle of the first slice.
     *
     * @param filter the kernel to rotate.
     * @param nrOfRotations the number of rotations.
     * @param minAngle the start angle, first slice included.
     * @param maxAngle the end angle.
     * @param output the output of the rotation.
     */
    @Override
    public void rotateKernels(imatrix filter, int nrOfRotations, float minAngle, float maxAngle, imatrix output) {
        fmatrix sincos = new fmatrix(2, nrOfRotations);
        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);
        float angle = minAngle;
        for (int i = 0; i < nrOfRotations; ++i) {
            float s = (float) Math.sin(angle);
            float c = (float) Math.cos(angle);
            sincos.set(0, i, s);
            sincos.set(1, i, c);
            angle += angleStep;
        }

        float2 mask = new float2();

        for (int oSlice = 0; oSlice < output.getNrOfSlices(); ++oSlice) {
            int rot = oSlice % nrOfRotations;
            int baseSlice = oSlice / nrOfRotations;

            float sa = (float) sincos.get(0, rot);
            float ca = (float) sincos.get(1, rot);
            float rcx = filter.getNrOfColumns() / 2.0f;
            float rcy = filter.getNrOfRows() / 2.0f;

            for (int x = 0; x < filter.getNrOfColumns(); ++x) {
                for (int y = 0; y < filter.getNrOfRows(); ++y) {
                    float rx = x - rcx;
                    float ry = y - rcy;

                    float ox = rcx + rx * ca - ry * sa;
                    float oy = rcy + rx * sa + ry * ca;

                    float xPerc = Math.abs(ox % 1);
                    float yPerc = Math.abs(oy % 1);
                    int startx = (int) ox;
                    int starty = (int) oy;

                    getKernelValue(filter, startx, starty, baseSlice, mask);
                    float i1 = mask.x;
                    float m1 = mask.y;
                    getKernelValue(filter, startx, starty + 1, baseSlice, mask);
                    float i2 = mask.x;
                    float m2 = mask.y;
                    getKernelValue(filter, startx + 1, starty, baseSlice, mask);
                    float i3 = mask.x;
                    float m3 = mask.y;
                    getKernelValue(filter, startx + 1, starty + 1, baseSlice, mask);
                    float i4 = mask.x;
                    float m4 = mask.y;

                    float a1 = (1 - xPerc) * (1 - yPerc);
                    float a2 = (1 - xPerc) * (yPerc);
                    float a3 = xPerc * (1 - yPerc);
                    float a4 = xPerc * yPerc;

                    float norm = a1 * m1 + a2 * m2 + a3 * m3 + a4 * m4;
                    float value = a1 * i1 + a2 * i2 + a3 * i3 + a4 * i4;
                    output.set(y, x, oSlice, Math.abs(norm) < 0.00001f ? value : value / norm);
                }
            }
        }
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
        fmatrix sincos = new fmatrix(2, nrOfRotations);
        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);
        float angle = minAngle;
        for (int i = 0; i < nrOfRotations; ++i) {
            float s = (float) Math.sin(-angle);
            float c = (float) Math.cos(-angle);
            sincos.set(0, i, s);
            sincos.set(1, i, c);
            angle += angleStep;
        }

        float2 mask = new float2();

        for (int oSlice = 0; oSlice < kernelOutput.getNrOfSlices(); ++oSlice) {
            for (int rot = 0; rot < nrOfRotations; ++rot) {
                int inputSlice = oSlice * nrOfRotations + rot;

                float sa = (float) sincos.get(0, rot);
                float ca = (float) sincos.get(1, rot);
                float rcx = kernelOutput.getNrOfColumns() / 2.0f;
                float rcy = kernelOutput.getNrOfRows() / 2.0f;

                for (int x = 0; x < kernelOutput.getNrOfColumns(); ++x) {
                    for (int y = 0; y < kernelOutput.getNrOfRows(); ++y) {
                        float rx = x - rcx;
                        float ry = y - rcy;

                        float ox = rcx + rx * ca - ry * sa;
                        float oy = rcy + rx * sa + ry * ca;

                        float xPerc = Math.abs(ox % 1);
                        float yPerc = Math.abs(oy % 1);
                        int startx = (int) ox;
                        int starty = (int) oy;

                        getKernelValue(rotatedOutput, startx, starty, inputSlice, mask);
                        float i1 = mask.x;
                        float m1 = mask.y;
                        getKernelValue(rotatedOutput, startx, starty + 1, inputSlice, mask);
                        float i2 = mask.x;
                        float m2 = mask.y;
                        getKernelValue(rotatedOutput, startx + 1, starty, inputSlice, mask);
                        float i3 = mask.x;
                        float m3 = mask.y;
                        getKernelValue(rotatedOutput, startx + 1, starty + 1, inputSlice, mask);
                        float i4 = mask.x;
                        float m4 = mask.y;

                        float a1 = (1 - xPerc) * (1 - yPerc);
                        float a2 = (1 - xPerc) * (yPerc);
                        float a3 = xPerc * (1 - yPerc);
                        float a4 = xPerc * yPerc;

                        float norm = a1 * m1 + a2 * m2 + a3 * m3 + a4 * m4;
                        float value = a1 * i1 + a2 * i2 + a3 * i3 + a4 * i4;

                        float result = Math.abs(norm) < 0.00001f ? value : value / norm;
                        float current = kernelOutput.get(y, x, oSlice);
                        kernelOutput.set(y, x, oSlice, current + result);
                    }
                }
            }
        }
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
     * @param output the output of this function.
     */
    @Override
    public void maxRotation(imatrix input, int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix output,
            imatrix rotOutput
    ) {
        int spf = output.getNrOfSlices() / nrOfFeatures;
        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);
        for (int h = 0; h < output.getNrOfHyperSlices(); ++h) {
            for (int s = 0; s < output.getNrOfSlices(); s += 1) {

                int subSlice = s % spf;
                int feature = s / spf;

                int inputSliceBase = feature * spf * nrOfRotations;
                for (int x = 0; x < output.getNrOfColumns(); ++x) {
                    for (int y = 0; y < output.getNrOfRows(); ++y) {
                        float max = -Float.MAX_VALUE;
                        float rot = 0;
                        for (int r = 0; r < nrOfRotations; ++r) {
                            float value = input.get(y, x, inputSliceBase + r * spf + subSlice, h);
                            if (value > max) {
                                max = value;
                                rot = r * angleStep;
                            }
                        }
                        rotOutput.set(y, x, s, h, rot);
                        output.set(y, x, s, h, max);
                    }
                }
            }
        }
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
    public void maxInverseRotation(imatrix valInput, imatrix rotInput,
            int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix output
    ) {
        int spf = valInput.getNrOfSlices() / nrOfFeatures;
        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);
        for (int h = 0; h < valInput.getNrOfHyperSlices(); ++h) {
            for (int s = 0; s < valInput.getNrOfSlices(); s += 1) {
                int subSlice = s % spf;
                int feature = s / spf;

                int inputSliceBase = feature * spf * nrOfRotations;
                for (int x = 0; x < valInput.getNrOfColumns(); ++x) {
                    for (int y = 0; y < valInput.getNrOfRows(); ++y) {

                        float rotValue = rotInput.get(y, x, s, h);
                        float value = valInput.get(y, x, s, h);

                        int rotIndex = Math.round(rotValue / angleStep);
                        int slice = inputSliceBase + rotIndex * spf + subSlice;
                        output.set(y, x, slice, h, value);
                    }
                }
            }
        }
    }

    private void getKernelValue(imatrix filter, int x, int y, int slice, float2 result) {
        if (x >= 0 && y >= 0 && x < filter.getNrOfColumns() && y < filter.getNrOfRows()) {
            result.x = filter.get(y, x, slice);
            result.y = 1;
        } else {
            result.x = 0;
            result.y = 0;
        }
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
        } else {
            copyInto(src, dst);
        }
    }

    /**
     * Copies the slices of matrix1 and matrix2 into the destination matrix. One
     * slice of the destination matrix will be composed of the concatenation of
     * a slice of the first matrix and a slice of the second matrix.
     *
     * @param matrix1 the first matrix.
     * @param matrix2 the second matrix.
     * @param dst the destination matrix.
     */
    @Override
    public void zip(imatrix matrix1, imatrix matrix2, imatrix dst) {
        int hSlices = Math.min(matrix1.getNrOfHyperSlices(), matrix2.getNrOfHyperSlices());

        hSlices = Math.min(hSlices, dst.getNrOfHyperSlices());
        for (int h = 0; h < hSlices; ++h) {
            for (int s = 0; s < matrix1.getNrOfSlices(); ++s) {
                for (int r = 0; r < matrix1.getNrOfRows(); ++r) {
                    for (int c = 0; c < matrix1.getNrOfColumns(); ++c) {
                        float v1 = matrix1.get(r, c, s, h);
                        dst.set(r, c, s * 2, h, v1);

                        float v2 = matrix2.get(r, c, s, h);
                        dst.set(r, c, s * 2 + 1, h, v2);
                    }
                }
            }
        }
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
        int hSlices = Math.min(dest1.getNrOfHyperSlices(), dest2.getNrOfHyperSlices());
        hSlices = Math.min(hSlices, src.getNrOfHyperSlices());
        for (int h = 0; h < hSlices; ++h) {
            for (int s = 0; s < src.getNrOfSlices(); ++s) {
                for (int r = 0; r < src.getNrOfRows(); ++r) {
                    for (int c = 0; c < src.getNrOfColumns(); ++c) {
                        if (s % 2 == 0) {
                            float v1 = src.get(r, c, s, h);
                            dest1.set(r, c, s / 2, h, v1);
                        } else {
                            float v2 = src.get(r, c, s, h);
                            dest2.set(r, c, s / 2, h, v2);
                        }
                    }
                }
            }
        }
    }

    @Override
    public void reset(fmatrix m) {
        m.applyFunction(x -> 0);
    }
}
