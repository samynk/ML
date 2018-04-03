/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.io;

import dae.matrix.imatrix;
import dae.neuralnet.ConvolutionLayer;
import dae.neuralnet.DeepLayer;
import dae.neuralnet.FuzzyficationLayer;
import dae.neuralnet.ILayer;
import dae.neuralnet.Layer;
import dae.neuralnet.LearningRate;
import dae.neuralnet.LearningRateConst;
import dae.neuralnet.LearningRateDecay;
import dae.neuralnet.PoolLayer;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.cost.CostFunction;
import dae.neuralnet.cost.CrossEntropyCostFunction;
import dae.neuralnet.cost.QuadraticCostFunction;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import static java.nio.file.StandardOpenOption.CREATE;
import java.util.Calendar;
import java.util.logging.Level;
import java.util.logging.Logger;

import static dae.neuralnet.io.DeepLayerBinaryID.*;
import java.io.DataOutputStream;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DeepLayerWriter {

    public void writeDeepLayer(Path origin, boolean increment, DeepLayer dl) {
        if (increment) {
            String fileName = origin.getFileName().toString();
            int firstIndex = -1;
            int startIndex = fileName.lastIndexOf(".");
            for (int i = startIndex - 1; i >= 0; --i) {
                char c = fileName.charAt(i);
                if (!Character.isDigit(c)) {
                    firstIndex = i + 1;
                    break;
                }
            }
            int number = 1;
            if (firstIndex < startIndex) {
                number = Integer.parseInt(fileName.substring(firstIndex,startIndex));
            }
            String newFileName = fileName.substring(0, firstIndex) + (number + 1) + ".nn";
            Path newPath = origin.resolveSibling(newFileName);
            writeDeepLayer(newPath, dl);
        } else {
            writeDeepLayer(origin, dl);
        }
    }

    public void writeDeepLayer(Path path, DeepLayer dl) {
        try {
            OutputStream os = Files.newOutputStream(path, CREATE);
            try (BufferedOutputStream bos = new BufferedOutputStream(os)) {
                DataOutputStream dos = new DataOutputStream(bos);
                writeHeader(dos, dl);
                writeLearningRate(dos, dl);
                writeCostFunction(dos, dl);

                writeLayers(dos, dl);
            }
        } catch (IOException ex) {
            Logger.getLogger(DeepLayerWriter.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void writeHeader(DataOutputStream dos, DeepLayer dl) throws IOException {
        Calendar c = Calendar.getInstance();
        dos.writeInt(HEADER);
        dos.writeInt(DATE);
        dos.writeShort((short) (c.get(Calendar.DAY_OF_MONTH)));
        dos.writeShort((short) (c.get(Calendar.MONTH)));
        dos.writeInt(c.get(Calendar.YEAR));

        dos.writeInt(TIME);
        dos.writeShort((short) c.get(Calendar.HOUR_OF_DAY));
        dos.writeShort((short) c.get(Calendar.MINUTE));

        DeepLayerMetaData dlmd = dl.getMetaData();
        String author = dlmd.getAuthor();
        if (author == null || author.length() == 0) {
            author = System.getProperty("user.name");
        }
        dos.writeInt(AUTHOR);
        dos.writeUTF(author);
    }

    private void writeLearningRate(DataOutputStream dos, DeepLayer dl) throws IOException {
        LearningRate lr = dl.getLearningRate();
        dos.writeInt(LEARNINGRATE);
        if (lr instanceof LearningRateConst) {
            LearningRateConst lrc = (LearningRateConst) lr;
            dos.writeInt(LEARNINGRATECONST);
            dos.writeFloat(lrc.getLearningRate(0));
        } else if (lr instanceof LearningRateDecay) {
            LearningRateDecay lrd = (LearningRateDecay) lr;
            dos.writeInt(LEARNINGRATEDECAY);
            dos.writeFloat(lrd.getBaseLearningRate());
            dos.writeFloat(lrd.getDecay());
        }
    }

    private void writeCostFunction(DataOutputStream dos, DeepLayer dl) throws IOException {
        CostFunction cf = dl.getCostFunction();
        dos.writeInt(COSTFUNCTION);
        if (cf instanceof QuadraticCostFunction) {
            dos.writeInt(COSTFUNCTIONQUADRATIC);
        } else if (cf instanceof CrossEntropyCostFunction) {
            dos.writeInt(COSTFUNCTIONCROSSENTROPY);
        }
    }

    private void writeLayers(DataOutputStream dos, DeepLayer dl) throws IOException {
        int i = 0;
        for (ILayer l : dl.getLayers()) {
            writeLayer(dos, l, i);
        }
    }

    private void writeLayer(DataOutputStream dos, ILayer l, int i) throws IOException {
        if (l instanceof Layer) {
            writeLayer(dos, (Layer) l);
        } else if (l instanceof ConvolutionLayer) {
            writeLayer(dos, (ConvolutionLayer) l);
        } else if (l instanceof PoolLayer) {
            writeLayer(dos, (PoolLayer) l);
        } else if (l instanceof FuzzyficationLayer) {
            writeLayer(dos, (FuzzyficationLayer) l);
        }
    }

    private void writeLayer(DataOutputStream dos, Layer l) throws IOException {
        dos.writeInt(LAYERNN);
        // nrofblocks
        if (l.isDropRateSet()) {
            dos.writeInt(8);
        } else {
            dos.writeInt(7);
        }
        dos.writeInt(LAYERNAME);
        dos.writeUTF(l.getName()!=null?l.getName():"");
        dos.writeInt(LAYERINPUTS);
        dos.writeInt(l.getNrOfInputs());
        dos.writeInt(LAYERBIASES);
        dos.writeInt(l.getNrOfBiases());
        dos.writeInt(LAYEROUTPUTS);
        dos.writeInt(l.getNrOfOutputs());
        dos.writeInt(LAYERBATCHSIZE);
        dos.writeInt(l.getBatchSize());
        if (l.isDropRateSet()) {
            dos.writeInt(LAYERDROPRATE);
            dos.writeFloat(l.getDropRate());
        }
        dos.writeInt(ACTIVATIONFUNCTION);
        ActivationFunction af = l.getActivationFunction();
        writeActivationFunction(af, dos);
        imatrix weights = l.getWeights();
        dos.writeInt(LAYERWEIGHTS);
        writeMatrix(dos, weights);
    }

    private void writeActivationFunction(ActivationFunction af, DataOutputStream dos) throws IOException {
        switch (af) {
            case CESIGMOID:
                dos.writeInt(ACTIVATIONFUNCTIONCESIGMOID);
                break;
            case SIGMOID:
                dos.writeInt(ACTIVATIONFUNCTIONSIGMOID);
                break;
            case IDENTITY:
                dos.writeInt(ACTIVATIONFUNCTIONIDENTITY);
                break;
            case RELU:
                dos.writeInt(ACTIVATIONFUNCTIONRELU);
                break;
            case LEAKYRELU:
                dos.writeInt(ACTIVATIONFUNCTIONLEAKYRELU);
                break;
            case SOFTMAX:
                dos.writeInt(ACTIVATIONFUNCTIONSOFMTAX);
                break;
            case TANH:
                dos.writeInt(ACTIVATIONFUNCTIONTANH);
                break;
        }
    }

    private void writeMatrix(DataOutputStream dos, imatrix m) throws IOException {
        dos.writeInt(MATRIX);
        // nrOfBlocks
        dos.writeInt(4);
        dos.writeInt(MATRIXROWS);
        dos.writeInt(m.getNrOfRows());
        dos.writeInt(MATRIXCOLUMNS);
        dos.writeInt(m.getNrOfColumns());
        dos.writeInt(MATRIXSLICES);
        dos.writeInt(m.getNrOfSlices());
        dos.writeInt(MATRIXDATAFLOAT);
        // general way of storing a matrix in row major order.
        for (int slice = 0; slice < m.getNrOfSlices(); ++slice) {
            for (int column = 0; column < m.getNrOfColumns(); ++column) {
                for (int row = 0; row < m.getNrOfRows(); ++row) {
                    float value = m.get(row, column, slice);
                    dos.writeFloat(value);
                }
            }
        }
    }

    private void writeLayer(DataOutputStream dos, ConvolutionLayer l) throws IOException {
        dos.writeInt(LAYERCONVOLUTION);
        // nrofblocks
        dos.writeInt(8);
        dos.writeInt(LAYERNAME);
        dos.writeUTF(l.getName()!=null?l.getName():"convolution");
        dos.writeInt(LAYERINPUTDIMENSION);
        dos.writeInt(l.getNrOfWInputs());
        dos.writeInt(l.getNrOfHInputs());
        dos.writeInt(l.getNrOfSInputs());
        dos.writeInt(LAYERBATCHSIZE);
        dos.writeInt(l.getBatchSize());
        dos.writeInt(LAYERFEATURES);
        dos.writeInt(l.getNrOfFeatures());
        dos.writeInt(LAYERFILTERSIZE);
        dos.writeInt(l.getFilterSize());
        dos.writeInt(LAYERFILTERSTRIDE);
        dos.writeInt(l.getFilterStride());
        dos.writeInt(ACTIVATIONFUNCTION);
        writeActivationFunction(l.getActivationFunction(), dos);
        dos.writeInt(LAYERWEIGHTS);
        writeMatrix(dos, l.getWeights());
    }

    private void writeLayer(DataOutputStream dos, PoolLayer l) throws IOException {
        dos.writeInt(LAYERMAXPOOL);
        // nrofblocks
        dos.writeInt(4);
        dos.writeInt(LAYERNAME);
        dos.writeUTF(l.getName());
        dos.writeInt(LAYERINPUTDIMENSION);
        dos.writeInt(l.getNrofWInputs());
        dos.writeInt(l.getNrOfHInputs());
        dos.writeInt(l.getNrOfSInputs());
        dos.writeInt(LAYERBATCHSIZE);
        dos.writeInt(l.getBatchSize());

        dos.writeInt(LAYERFILTERSIZEX);
        dos.writeInt(l.getScaleX());
        dos.writeInt(LAYERFILTERSIZEY);
        dos.writeInt(l.getScaleY());

    }

    private void writeLayer(DataOutputStream dos, FuzzyficationLayer l) throws IOException {
        dos.writeInt(LAYERFUZZY);
        // nrofblocks
        dos.writeInt(7);
        dos.writeInt(LAYERNAME);
        dos.writeUTF(l.getName());

        dos.writeInt(LAYERINPUTS);
        dos.writeInt(l.getNrOfInputs());
        dos.writeInt(LAYERFUZZYCLASSES);
        dos.writeInt(l.getNrOfClasses());
        dos.writeInt(LAYERBATCHSIZE);
        dos.writeInt(l.getBatchSize());
        dos.writeInt(ACTIVATIONFUNCTION);
        writeActivationFunction(l.getActivationFunction(), dos);
        dos.writeInt(LAYERWEIGHTSA);
        writeMatrix(dos, l.getAWeights());
        dos.writeInt(LAYERWEIGHTSB);
        writeMatrix(dos, l.getBWeights());
    }

}
