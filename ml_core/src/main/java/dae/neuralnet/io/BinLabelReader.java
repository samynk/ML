package dae.neuralnet.io;

import dae.matrix.fmatrix;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Koen
 */
public class BinLabelReader {

    private String filename;
    private fmatrix result;

    private byte[] intBlock = new byte[4];

    public BinLabelReader(String filename) {
        this.filename = filename;
        Read();
    }

    private void Read() {
        try {
            BufferedInputStream bis = new BufferedInputStream(getClass().getResourceAsStream(filename));
            
            DataInputStream dis = new DataInputStream(bis);
            int magicNumber = ReadInt(bis);
            int nrOfItems = ReadInt(bis);

            result = new fmatrix(1, nrOfItems);

            byte[] pixelBlock = new byte[nrOfItems];
            dis.read(pixelBlock);
            for (int i = 0; i < nrOfItems; ++i) {
                float label = (pixelBlock[i] & 0xff);
                result.set(0, i, label);
            }
            dis.close();
        } catch (IOException ex) {
            Logger.getLogger(BinImageReader.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    private int ReadInt(BufferedInputStream bis) throws IOException {
        bis.read(intBlock);
        int value = ((intBlock[0] & 0xff) << 24);
        value += ((intBlock[1] & 0xff) << 16);
        value += ((intBlock[2] & 0xff) << 8);
        value += (intBlock[3] & 0xff);
        return value;
    }

    public fmatrix getResult() {
        return result;
    }
}
