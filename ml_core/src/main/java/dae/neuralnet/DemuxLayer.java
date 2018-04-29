/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.activation.ActivationFunction;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DemuxLayer implements ILayer {

    public String name;

    private final ArrayList<ILayer> prototype = new ArrayList<>();
    private final ILayer[][] layers;

    private final int mux;
    private final int numLayers;

    private final int numInputs;
    private final int numOutputs;
    private final int batchSize;

    private imatrix inputs;
    private final imatrix outputs;
    private final imatrix errors;

    private imatrix dupErrors;

    private final ArrayList<imatrix> muxOutputs = new ArrayList<>();
    private final ArrayList<imatrix> muxErrors = new ArrayList<>();

    public DemuxLayer(int multiplicity, int batchSize, ILayer... prototypes) {
        for (int i = 0; i < prototypes.length; ++i) {
            prototype.add(prototypes[i]);
        }

        layers = new ILayer[prototypes.length][multiplicity];
        mux = multiplicity;
        numLayers = prototypes.length;

        for (int muxi = 0; muxi < multiplicity; ++muxi) {
            for (int layer = 0; layer < numLayers; ++layer) {
                ILayer protoLayer = prototype.get(layer);
                layers[layer][muxi] = protoLayer.duplicate();
                layers[layer][muxi].setName(protoLayer.getName());
            }
        }

        int lastLayer = prototypes.length - 1;

        numInputs = prototypes[0].getNrOfInputs();
        numOutputs = prototypes[prototypes.length - 1].getNrOfOutputs() * multiplicity;
        this.batchSize = batchSize;

        outputs = new fmatrix(numOutputs, 1, 1, batchSize);
        errors = new fmatrix(numOutputs, 1, 1, batchSize);

        for (int muxi = 0; muxi < multiplicity; ++muxi) {
            this.muxErrors.add(layers[lastLayer][muxi].getErrors());
            this.muxOutputs.add(layers[lastLayer][muxi].getOutputs());
        }
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public int getNrOfInputs() {
        return numInputs;
    }

    @Override
    public int getNrOfOutputs() {
        return numOutputs;
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return ActivationFunction.IDENTITY;
    }

    @Override
    public void forward() {
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                layers[l][muxi].forward();
                if ((l + 1) < numLayers) {
                    layers[l + 1][muxi].setInputs(layers[l][muxi].getOutputs());
                }
            }
        }
        for( imatrix im : this.muxOutputs){
            im.sync();
        }
        fmatrix.zip(this.muxOutputs, outputs);
        outputs.sync();
    }

    @Override
    public void setInputs(imatrix input) {
        //fmatrix.copyIntoSlice(input, this.inputs);
        for (int i = 0; i < mux; ++i) {
            layers[0][i].setInputs(input);
        }
    }

    @Override
    public imatrix getInputs() {
        return inputs;
    }

    @Override
    public void setIdeal(imatrix ideals) {

    }

    @Override
    public imatrix getErrors() {
        return errors;
    }

    @Override
    public imatrix getOutputs() {
        return outputs;
    }

    @Override
    public void backpropagate(float learningRate) {
        fmatrix.unzip(errors, this.muxErrors);
        errors.sync();
        

        int lastLayer = numLayers - 1;
        for (int l = lastLayer; l >= 0; --l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                layers[l][muxi].backpropagate(learningRate);
                if (l - 1 >= 0) {
                    layers[l][muxi].calculateErrors(layers[l - 1][muxi].getErrors());
                }
            }
        }
    }

    @Override
    public void calculateNewWeights(float learningRate) {
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                layers[l][muxi].calculateNewWeights(learningRate);
            }
        }
    }

    @Override
    public void calculateErrors(imatrix errors) {
        // add the calculations of all the first layers together.
        errors.reset();
        if (dupErrors == null) {
            dupErrors = new fmatrix(errors.getNrOfRows(), errors.getNrOfColumns(), errors.getNrOfSlices(), errors.getNrOfHyperSlices(), errors.getZeroPadding());
        }
        for (int muxi = 0; muxi < mux; ++muxi) {
            layers[0][muxi].calculateErrors(dupErrors);
            fmatrix.dotadd(errors, errors, dupErrors);
        }
    }

    @Override
    public void adaptWeights(float factor) {
        calculateNewWeights(factor);
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                layers[l][muxi].adaptWeights(factor);
            }
        }
    }

    @Override
    public void randomizeWeights(Random r, float min, float max) {
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                layers[l][muxi].randomizeWeights(r, min, max);
            }
        }
    }

    @Override
    public void analyzeWeights() {
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                layers[l][muxi].analyzeWeights();
            }
        }
    }

    @Override
    public void writeWeightImage(String file) {
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                String n = layers[l][muxi].getName();
                layers[l][muxi].writeWeightImage(file + "_" + l + "_" + n + "_" + muxi);
            }
        }
    }

    @Override
    public void writeOutputImage(String file) {
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                String n = layers[l][muxi].getName();
                layers[l][muxi].writeOutputImage(file + "_" + n + "_" + muxi);
            }
        }
    }

    @Override
    public void sync() {
        for (int l = 0; l < numLayers; ++l) {
            for (int muxi = 0; muxi < mux; ++muxi) {
                layers[l][muxi].sync();
            }
        }
    }

    @Override
    public ILayer duplicate() {
        return new DemuxLayer(mux, batchSize, (ILayer[]) prototype.toArray());
    }
}
