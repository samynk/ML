/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;

/**
 * Kernels to calculate the activation functions.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class ActivationKernel extends OpenCLKernel {
    private cl_kernel softmax;
    private cl_kernel sigmoid;
    private cl_kernel dsigmoid;
    private cl_kernel relu;
    private cl_kernel drelu;
    private cl_kernel leakyrelu;
    private cl_kernel dleakyrelu;
    private cl_kernel tanh;
    private cl_kernel dtanh;

    private final static float LEAKYRELUCONST = 0.001f;
    
    

    /**
     * Creates a new convolution kernel.
     *
     */
    public ActivationKernel() {
        super("/kernels/activation.cl");
    }

    @Override
    public void init(cl_context context, cl_command_queue commandQueue) {
        super.init(context, commandQueue);
        softmax = this.createKernel("softmax");
        sigmoid = this.createKernel("sigmoid");
        dsigmoid = this.createKernel("dsigmoid");
        relu = this.createKernel("relu");
        drelu = this.createKernel("drelu");
        leakyrelu = this.createKernel("leakyrelu");
        dleakyrelu = this.createKernel("dleakyrelu");
        tanh = this.createKernel("a_tanh");
        dtanh = this.createKernel("dtanh");
        super.releaseProgram();
    }

    public void softmax(imatrix O){
        applyKernel(softmax, O);
    }
    
    public void sigmoid(imatrix O) {
        applyKernel(sigmoid, O);
    }

    public void dsigmoid(imatrix O) {
        applyKernel(dsigmoid, O);
    }

    public void relu(imatrix O) {
        applyKernel(relu, O);
    }

    public void drelu(imatrix O) {
        applyKernel(drelu, O);
    }

    public void leakyrelu(imatrix O) {
        applyKernel(leakyrelu, O, LEAKYRELUCONST);
    }

    public void dleakyrelu(imatrix O) {
        applyKernel(dleakyrelu, O, LEAKYRELUCONST);
    }

    public void tanh(imatrix O) {
        applyKernel(tanh, O);
    }

    public void dtanh(imatrix O) {
        applyKernel(dtanh, O);
    }
    
    public void didentity(imatrix O) {
        GPU.fillR(O,1.0f);
        O.getDeviceBuffer().markRMatrixAsMaster();
    }
}
