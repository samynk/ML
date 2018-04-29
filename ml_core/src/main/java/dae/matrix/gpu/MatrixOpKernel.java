/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clSetKernelArg;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class MatrixOpKernel extends OpenCLKernel {

    cl_kernel dotadd;
    cl_kernel dotaddlc;
    cl_kernel sumPerRow;
    cl_kernel dotsubtract;
    cl_kernel dotmultiply;
    cl_kernel dotmultiplyfactor;
    cl_kernel square;
    cl_kernel root;
    cl_kernel init_random;
    cl_kernel random;
    cl_kernel adamVelocity;
    cl_kernel adamAdaptWeights;
    cl_kernel sumPerSlice;

    private long[] localWorkSize = new long[]{OpenCLKernel.DEFAULTWORKSIZE};

    /**
     * Creates a new MatrixOpKernel object.
     */
    public MatrixOpKernel() {
        super("/kernels/neuralnet/matrixop.cl");
        seedMatrix = new intmatrix(1, 1);
    }

    @Override
    public void init(cl_context context, cl_command_queue commandQueue) {
        super.init(context, commandQueue);
        dotadd = this.createKernel("dotadd");
        dotaddlc = this.createKernel("dotaddlc");
        sumPerRow = this.createKernel("sumPerRow");
        dotsubtract = this.createKernel("dotsubtract");
        dotmultiply = this.createKernel("dotmultiply");
        dotmultiplyfactor = this.createKernel("dotmultiplyfactor");
        square = this.createKernel("squared");
        root = this.createKernel("root");
        init_random = this.createKernel("rnd_init");
        random = this.createKernel("rnd_1");
        adamVelocity = this.createKernel("adamVelocity");
        adamAdaptWeights = this.createKernel("adamAdaptWeights");
        sumPerSlice = this.createKernel("sumPerSlice");

        super.releaseProgram();
    }

    public imatrix dotadd(imatrix O, imatrix op1, imatrix op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        cl_mem memOutput = oDB.getMem();
        cl_mem mem_op1 = op1.getDeviceBuffer().upload();
        cl_mem mem_op2 = op2.getDeviceBuffer().upload();

        clSetKernelArg(dotadd, 0, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotadd, 1, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotadd, 2, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                dotadd,
                1,
                null,
                oDB.getGlobalWorkSize(),
                localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
        return O;
    }

    public imatrix dotadd(imatrix O, float factor1, imatrix op1, float factor2, imatrix op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        float[] factors = new float[]{factor1, factor2};
        cl_mem memOutput = oDB.getMem();

        cl_mem mem_op1 = op1.getDeviceBuffer().upload();
        cl_mem mem_op2 = op2.getDeviceBuffer().upload();

        clSetKernelArg(dotaddlc, 0, Sizeof.cl_float2, Pointer.to(factors));
        clSetKernelArg(dotaddlc, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotaddlc, 2, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotaddlc, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                dotaddlc,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
        return O;
    }

    public void sumPerRow(imatrix input, imatrix output) {
        FloatDeviceBuffer oDB = output.getDeviceBuffer();
        FloatDeviceBuffer iDB = input.getDeviceBuffer();
        int[] h = new int[]{input.getNrOfHyperSlices(), input.getHyperSliceSize()};
        cl_mem memOutput = oDB.getMem();
        GPU.zeroFill(output);
        cl_mem mem_op1 = input.getDeviceBuffer().upload();

        clSetKernelArg(sumPerRow, 0, Sizeof.cl_int2, Pointer.to(h));
        clSetKernelArg(sumPerRow, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(sumPerRow, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clEnqueueNDRangeKernel(
                commandQueue,
                sumPerRow,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
    }

    public imatrix adamVelocity(imatrix O, float beta2, imatrix previousVelocity, imatrix currentGradient) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        float[] factors = new float[]{beta2};
        cl_mem memOutput = oDB.getMem();

        cl_mem mem_op1 = previousVelocity.getDeviceBuffer().upload();
        cl_mem mem_op2 = currentGradient.getDeviceBuffer().upload();

        clSetKernelArg(adamVelocity, 0, Sizeof.cl_float, Pointer.to(factors));
        clSetKernelArg(adamVelocity, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(adamVelocity, 2, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(adamVelocity, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                adamVelocity,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
        return O;
    }

    public imatrix adamAdaptWeights(imatrix weights, float eta, float beta1, float beta2, float epsilon, imatrix moment, imatrix velocity) {
        FloatDeviceBuffer oDB = weights.getDeviceBuffer();
        float[] factors = new float[]{eta, 1f / (1 - beta1), 1f / (1 - beta2), epsilon};
        cl_mem memOutput = oDB.upload();

        cl_mem mem_op1 = moment.getDeviceBuffer().upload();
        cl_mem mem_op2 = velocity.getDeviceBuffer().upload();

        clSetKernelArg(adamAdaptWeights, 0, Sizeof.cl_float4, Pointer.to(factors));
        clSetKernelArg(adamAdaptWeights, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(adamAdaptWeights, 2, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(adamAdaptWeights, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                adamAdaptWeights,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
        return weights;
    }

    public imatrix dotsubtract(imatrix O, imatrix op1, imatrix op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        cl_mem memOutput = oDB.getMem();

        cl_mem mem_op1 = op1.getDeviceBuffer().upload();
        cl_mem mem_op2 = op2.getDeviceBuffer().upload();

        clSetKernelArg(dotsubtract, 0, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotsubtract, 1, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotsubtract, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clEnqueueNDRangeKernel(
                commandQueue,
                dotsubtract,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
        return O;
    }

    public imatrix dotmultiply(imatrix O, imatrix op1, imatrix op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        cl_mem memOutput = oDB.getMem();

        cl_mem mem_op1 = op1.getDeviceBuffer().upload();
        cl_mem mem_op2 = op2.getDeviceBuffer().upload();

        clSetKernelArg(dotmultiply, 0, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotmultiply, 1, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotmultiply, 2, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                dotmultiply,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
        return O;
    }

    public imatrix dotmultiply(imatrix O, imatrix op1, float op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        cl_mem memOutput = oDB.getMem();

        cl_mem mem_op1 = op1.getDeviceBuffer().upload();

        clSetKernelArg(dotmultiplyfactor, 0, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotmultiplyfactor, 1, Sizeof.cl_float, Pointer.to(new float[]{op2}));
        clSetKernelArg(dotmultiplyfactor, 2, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                dotmultiplyfactor,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markGpuAsMaster();
        return O;
    }

    public imatrix square(imatrix m) {
        super.applyKernel(square, m);
        return m;
    }

    public imatrix root(imatrix m) {
        super.applyKernel(root, m);
        return m;
    }
    
    public void sumPerSlice(imatrix src, imatrix dst){
        FloatDeviceBuffer srcDB = src.getDeviceBuffer();
        cl_mem srcMem = srcDB.upload();
        
        FloatDeviceBuffer dstDB = dst.getDeviceBuffer();
        cl_mem dstMem = dstDB.getMem();
        
        clSetKernelArg(sumPerSlice, 0, Sizeof.cl_mem, Pointer.to(srcMem));
        clSetKernelArg(sumPerSlice, 1, Sizeof.cl_int4, Pointer.to(srcDB.getDimensionSizes()));
        clSetKernelArg(sumPerSlice, 2, Sizeof.cl_mem, Pointer.to(dstMem));
        clSetKernelArg(sumPerSlice, 3, Sizeof.cl_int4, Pointer.to(dstDB.getDimensionSizes()));
        clSetKernelArg(sumPerSlice, 4, Sizeof.cl_int, Pointer.to(new int[]{dst.getSize()}));

        clEnqueueNDRangeKernel(
                commandQueue,
                sumPerSlice,
                1,
                null,
                dstDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        dstDB.markGpuAsMaster();
    }

    private intmatrix seedMatrix;

    public void randomize(imatrix m, float min, float max) {
        if (m.getSize() > seedMatrix.getSize()) {
            seedMatrix = new intmatrix(m.getSize(), 1);
            applyKernel(init_random, seedMatrix);
        }
        cl_mem seedMem = seedMatrix.getDeviceBuffer().getRWMem();
        clSetKernelArg(random, 1, Sizeof.cl_mem, Pointer.to(seedMem));
        applyKernel(random, m);
    }
}
