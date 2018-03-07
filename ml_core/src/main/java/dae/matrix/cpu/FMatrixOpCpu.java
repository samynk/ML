/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.cpu;

import dae.matrix.gpu.GPU;
import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import dae.matrix.op.FMatrixOp;
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
     * @param output the output matrix.
     *
     */
    @Override
    public void batchBackpropMaxPool(imatrix input, intmatrix maskLayer, imatrix output) {
        int scaleX = output.getNrOfColumns() / input.getNrOfColumns();
        int scaleY = output.getNrOfRows() / input.getNrOfRows();

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
        for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
            for (int row = 0; row < result.getNrOfRows(); ++row) {
                for (int column = 0; column < result.getNrOfColumns(); ++column) {
                    float op1value = op1.get(row, column, slice);
                    float op2value = op2.get(row, column, slice);
                    result.set(row, column, slice, op1value + op2value);
                }
            }
        }
        return result;
    }
    
    /**
     * Calculates the element by element addition of 
     * factor1 * op1 and factor2 * op2.
     *
     * @param result the matrix to store the result.
     * @param factor1 the first factor.
     * @param op1 the first operand.
     * @param factor2 the second factor.
     * @param op2 the second operand.
     * @return the result matrix
     */
    @Override
    public imatrix dotadd(imatrix result, float factor1, imatrix op1, float factor2, imatrix op2){
         for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
            for (int row = 0; row < result.getNrOfRows(); ++row) {
                for (int column = 0; column < result.getNrOfColumns(); ++column) {
                    float op1value = op1.get(row, column, slice);
                    float op2value = op2.get(row, column, slice);
                    result.set(row, column, slice, factor1 * op1value + factor2 * op2value);
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
    public imatrix dotsubtract(imatrix result, imatrix op1, imatrix op2) {
        for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
            for (int row = 0; row < result.getNrOfRows(); ++row) {
                for (int column = 0; column < result.getNrOfColumns(); ++column) {
                    float op1value = op1.get(row, column, slice);
                    float op2value = op2.get(row, column, slice);
                    result.set(row, column, slice, op1value - op2value);
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
        for (int slice = 0; slice < result.getNrOfSlices(); ++slice) {
            for (int row = 0; row < result.getNrOfRows(); ++row) {
                for (int column = 0; column < result.getNrOfColumns(); ++column) {
                    float op1value = op1.get(row, column, slice);
                    float op2value = op2.get(row, column, slice);
                    result.set(row, column, slice, op1value * op2value);
                }
            }
        }
        return result;
    }

}
