/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.io;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DeepLayerBinaryID {

    public static final int HEADER = 0x00DAE0A1;
    public static final short DATE = 0;
    public static final short TIME = 1;
    public static final short AUTHOR = 2;

    public static final int LEARNINGRATE = 100;
    public static final int LEARNINGRATECONST = 101;
    public static final int LEARNINGRATEDECAY = 102;

    public static final int COSTFUNCTION = 200;
    public static final int COSTFUNCTIONQUADRATIC = 201;
    public static final int COSTFUNCTIONCROSSENTROPY = 202;

    public static final int LAYERNN = 300;
    public static final int LAYERCONVOLUTION = 301;
    public static final int LAYERMAXPOOL = 302;
    public static final int LAYERFUZZY = 303;

    public static final int LAYERNAME = 400;
    public static final int LAYERINPUTS = 401;
    public static final int LAYEROUTPUTS = 402;
    public static final int LAYERBIASES = 403;
    public static final int LAYERDROPRATE = 404;
    public static final int LAYERWEIGHTS = 405;
    public static final int LAYERWEIGHTSA = 406;
    public static final int LAYERWEIGHTSB = 407;
    public static final int LAYERINPUTDIMENSION = 408;
    public static final int LAYERFEATURES = 409;
    public static final int LAYERFILTERSIZE = 410;
    public static final int LAYERFILTERSTRIDE = 411;
    public static final int LAYERFILTERSIZEX = 412;
    public static final int LAYERFILTERSIZEY = 413;
    public static final int LAYERFUZZYCLASSES = 414;
    public static final int LAYERBATCHSIZE = 415;

    public static final int ACTIVATIONFUNCTION = 500;
    public static final int ACTIVATIONFUNCTIONTANH = 501;
    public static final int ACTIVATIONFUNCTIONSOFMTAX = 502;
    public static final int ACTIVATIONFUNCTIONIDENTITY = 503;
    public static final int ACTIVATIONFUNCTIONSIGMOID = 504;
    public static final int ACTIVATIONFUNCTIONCESIGMOID = 505;
    public static final int ACTIVATIONFUNCTIONLEAKYRELU = 506;
    public static final int ACTIVATIONFUNCTIONRELU = 507;

    public static final int MATRIX = 600;

    public static final int MATRIXROWS = 601;
    public static final int MATRIXCOLUMNS = 602;
    public static final int MATRIXSLICES = 603;

    public static final int MATRIXDATAINT = 610;
    public static final int MATRIXDATALONG = 611;
    public static final int MATRIXDATABOOLEAN = 612;
    public static final int MATRIXDATAFLOAT = 613;
    public static final int MATRIXDATADOUBLE = 614;

}
