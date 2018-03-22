/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.io;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.neuralnet.ConvolutionLayer;
import dae.neuralnet.DeepLayer;
import dae.neuralnet.FuzzyficationLayer;
import dae.neuralnet.ILayer;
import dae.neuralnet.Layer;
import dae.neuralnet.LearningRateConst;
import dae.neuralnet.LearningRateDecay;
import dae.neuralnet.PoolLayer;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.cost.CrossEntropyCostFunction;
import dae.neuralnet.cost.QuadraticCostFunction;
import static dae.neuralnet.io.DeepLayerBinaryID.*;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DeepLayerReader {

    public DeepLayer readDeepLayer(Path path) {
        try {
            InputStream os = Files.newInputStream(path);
            try (BufferedInputStream bis = new BufferedInputStream(os)) {
                DataInputStream dis = new DataInputStream(bis);
                DeepLayer result = new DeepLayer();
                readHeader(dis, result);
                readFunctions(dis, result);
                readLayers(dis, result);
                result.getMetaData().setPath(path);
                return result;
            }
        } catch (IOException ex) {
            Logger.getLogger(DeepLayerWriter.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

    private void readHeader(DataInputStream dis, DeepLayer dl) throws IOException {
        DeepLayerMetaData dlmd = new DeepLayerMetaData();
        Calendar c = Calendar.getInstance();

        int header = dis.readInt();
        System.out.println("header:" + Integer.toHexString(header));

        // creation time.
        int dateBlock = dis.readInt();
        short dayOfMonth = dis.readShort();
        short month = dis.readShort();
        int year = dis.readInt();

        int timeBlock = dis.readInt();
        short hour = dis.readShort();
        short minute = dis.readShort();

        c.set(year, month, dayOfMonth, hour, minute);
        dlmd.setCreationTime(c);

        int block = dis.readInt();
        switch (block) {
            case AUTHOR:
                readAuthor(dis, dlmd);
                break;
        }
        dl.setMetaData(dlmd);
    }

    private void readAuthor(DataInputStream dis, DeepLayerMetaData dlmd) throws IOException {
        String author = dis.readUTF();
        dlmd.setAuthor(author);
    }

    private void readFunctions(DataInputStream dis, DeepLayer result) throws IOException {
        int block = dis.readInt();
        readFunctionBlock(dis, block, result);
        block = dis.readInt();
        readFunctionBlock(dis, block, result);
    }

    private void readFunctionBlock(DataInputStream dis, int block, DeepLayer dl) throws IOException {
        switch (block) {
            case COSTFUNCTION:
                int type = dis.readInt();
                switch (type) {
                    case COSTFUNCTIONCROSSENTROPY:
                        dl.setCostFunction(new CrossEntropyCostFunction());
                        break;
                    case COSTFUNCTIONQUADRATIC:
                        dl.setCostFunction(new QuadraticCostFunction());
                        break;
                }
                break;
            case LEARNINGRATE:
                int lrtype = dis.readInt();
                switch (lrtype) {
                    case LEARNINGRATECONST:
                        float learningRate = dis.readFloat();
                        dl.setLearningRate(new LearningRateConst(learningRate));
                        break;
                    case LEARNINGRATEDECAY:
                        float base = dis.readFloat();
                        float decay = dis.readFloat();
                        dl.setLearningRate(new LearningRateDecay(base, decay));
                        break;
                }
        }
    }

    private void readLayers(DataInputStream dis, DeepLayer result) throws IOException {
        ArrayList<ILayer> layers = new ArrayList<>();

        while (dis.available() > 0) {
            int layerType = dis.readInt();
            switch (layerType) {
                case LAYERNN:
                    ILayer layer = readLayerNN(dis);
                    layers.add(layer);
                    break;
                case LAYERCONVOLUTION:
                    ILayer lc = readLayerConvolution(dis);
                    layers.add(lc);
                    break;
                case LAYERMAXPOOL:
                    ILayer lp = readLayerMaxpool(dis);
                    layers.add(lp);
                    break;
                case LAYERFUZZY:
                    ILayer lf = readLayerFuzzy(dis);
                    layers.add(lf);
                    break;
            }
        }
        result.setLayers(layers);
    }

    private ILayer readLayerNN(DataInputStream dis) throws IOException {
        int nrOfBlocks = dis.readInt();
        String layerName = "nn";
        int inputs = 0, biases = 0, outputs = 0, batchSize =1;
        boolean dropRateSet = false;
        float dropRate = 0.0f;
        ActivationFunction function = ActivationFunction.IDENTITY;
        imatrix weights = null;

        for (int i = 0; i < nrOfBlocks; ++i) {
            int type = dis.readInt();
            switch (type) {
                case LAYERNAME:
                    layerName = dis.readUTF();
                    break;
                case LAYERINPUTS:
                    inputs = dis.readInt();
                    break;
                case LAYERBIASES:
                    biases = dis.readInt();
                    break;
                case LAYEROUTPUTS:
                    outputs = dis.readInt();
                    break;
                case LAYERBATCHSIZE:
                    batchSize = dis.readInt();
                    break;
                case LAYERDROPRATE:
                    dropRateSet = true;
                    dropRate = dis.readFloat();
                    break;
                case ACTIVATIONFUNCTION:
                    int afType = dis.readInt();
                    function = parseActivationFunction(afType);
                    break;
                case LAYERWEIGHTS:
                    weights = readMatrix(dis);
                    break;
            }
        }
        Layer l = new Layer(inputs, biases, outputs, batchSize, function, weights);
        l.setName(layerName);
        if (dropRateSet) {
            l.setDropRate(dropRate);
        }
        return l;
    }

    private ILayer readLayerConvolution(DataInputStream dis) throws IOException {
        int nrOfBlocks = dis.readInt();
        int wInputs = 0, hInputs = 0, sInputs = 0;
        int features = 0, filterSize = 0, filterStride = 0;
        int batchSize = 1;
        String layerName = "convolution";
        ActivationFunction function = ActivationFunction.IDENTITY;
        imatrix weights = null;

        for (int i = 0; i < nrOfBlocks; ++i) {
            int type = dis.readInt();
            switch (type) {
                case LAYERNAME:
                    layerName = dis.readUTF();
                    break;
                case LAYERINPUTDIMENSION:
                    wInputs = dis.readInt();
                    hInputs = dis.readInt();
                    sInputs = dis.readInt();
                    break;
                case LAYERBATCHSIZE:
                    batchSize = dis.readInt();
                    break;
                case LAYERFEATURES:
                    features = dis.readInt();
                    break;
                case LAYERFILTERSIZE:
                    filterSize = dis.readInt();
                    break;
                case LAYERFILTERSTRIDE:
                    filterStride = dis.readInt();
                    break;
                case ACTIVATIONFUNCTION:
                    int afType = dis.readInt();
                    function = parseActivationFunction(afType);
                    break;
                case LAYERWEIGHTS:
                    weights = readMatrix(dis);
                    break;
            }
        }
        ConvolutionLayer cl = new ConvolutionLayer(wInputs, hInputs, sInputs, features, filterSize, filterStride, batchSize, function, weights);
        cl.setName(layerName);
        return cl;
    }

    private ILayer readLayerMaxpool(DataInputStream dis) throws IOException {
        int nrOfBlocks = dis.readInt();
        int wInputs = 0, hInputs = 0, sInputs = 0;
        int scaleX = 0, scaleY = 0;
        int batchSize = 1;
        String layerName = "convolution";
        ActivationFunction function = ActivationFunction.IDENTITY;
        imatrix weights = null;

        for (int i = 0; i < nrOfBlocks; ++i) {
            int type = dis.readInt();
            switch (type) {
                case LAYERNAME:
                    layerName = dis.readUTF();
                    break;
                case LAYERINPUTDIMENSION:
                    wInputs = dis.readInt();
                    hInputs = dis.readInt();
                    sInputs = dis.readInt();
                    break;
                case LAYERBATCHSIZE:
                    batchSize = dis.readInt();
                    break;
                case LAYERFILTERSIZEX:
                    scaleX = dis.readInt();
                    break;
                case LAYERFILTERSIZEY:
                    scaleY = dis.readInt();
                    break;
            }
        }
        PoolLayer pl = new PoolLayer(wInputs, hInputs, sInputs, scaleX, scaleY, batchSize);
        pl.setName(layerName);
        return pl;
    }

    private FuzzyficationLayer readLayerFuzzy(DataInputStream dis) throws IOException {
        String layerName = "fuzzy";
        int inputs = 0;
        int classes = 0;
        int batchSize = 1;
        ActivationFunction function = ActivationFunction.IDENTITY;
        imatrix weightA = null;
        imatrix weightB = null;
        int nrOfBlocks = dis.readInt();
        for (int i = 0; i < nrOfBlocks; ++i) {
            int type = dis.readInt();
            switch (type) {
                case LAYERNAME:
                    layerName = dis.readUTF();
                    break;
                case LAYERINPUTS:
                    inputs = dis.readInt();
                    break;
                case LAYERBATCHSIZE:
                    batchSize = dis.readInt();
                    break;
                case LAYERFUZZYCLASSES:
                    classes = dis.readInt();
                    break;
                case ACTIVATIONFUNCTION:
                    int atype = dis.readInt();
                    function = parseActivationFunction(atype);
                    break;
                case LAYERWEIGHTSA:
                    weightA = readMatrix(dis);
                    break;
                case LAYERWEIGHTSB:
                    weightB = readMatrix(dis);
                    break;
            }
        }

        FuzzyficationLayer fl = new FuzzyficationLayer(inputs, classes, batchSize, weightA, weightB, function);
        fl.setName(layerName);
        return fl;
    }

    private ActivationFunction parseActivationFunction(int type) {
        switch (type) {
            case ACTIVATIONFUNCTIONIDENTITY:
                return ActivationFunction.IDENTITY;
            case ACTIVATIONFUNCTIONSIGMOID:
                return ActivationFunction.SIGMOID;
            case ACTIVATIONFUNCTIONCESIGMOID:
                return ActivationFunction.CESIGMOID;
            case ACTIVATIONFUNCTIONRELU:
                return ActivationFunction.RELU;
            case ACTIVATIONFUNCTIONLEAKYRELU:
                return ActivationFunction.LEAKYRELU;
            case ACTIVATIONFUNCTIONSOFMTAX:
                return ActivationFunction.SOFTMAX;
            case ACTIVATIONFUNCTIONTANH:
                return ActivationFunction.TANH;
            default:
                return ActivationFunction.IDENTITY;
        }
    }

    private imatrix readMatrix(DataInputStream dis) throws IOException {
        int matrix = dis.readInt();
        int nrOfBlocks = dis.readInt();

        int rows = 0;
        int columns = 0;
        // default
        int slices = 1;
        // read dimensions, last block should be matrix data.
        for (int i = 0; i < nrOfBlocks - 1; ++i) {
            int type = dis.readInt();
            switch (type) {
                case MATRIXROWS:
                    rows = dis.readInt();
                    break;
                case MATRIXCOLUMNS:
                    columns = dis.readInt();
                    break;
                case MATRIXSLICES:
                    slices = dis.readInt();
                    break;
            }
        }
        int dataType = dis.readInt();
        imatrix m = null;
        switch (dataType) {
            case MATRIXDATAFLOAT:
                m = readFloatMatrix(dis, rows, columns, slices);
                break;
        }

        return m;
    }

    private imatrix readFloatMatrix(DataInputStream dis, int rows, int columns, int slices) throws IOException {
        fmatrix matrix = new fmatrix(rows, columns, slices);
        matrix.getHostData().rewind();
        for (int i = 0; i < matrix.getSize(); ++i) {
            matrix.getHostData().put(dis.readFloat());
        }
        return matrix;
    }
}
