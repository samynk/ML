package dae.neuralnet.io;

import dae.matrix.fmatrix;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Koen
 */
public class BinImageReader {

    private final String filename;
    private fmatrix result;

    private byte[] intBlock = new byte[4];

    public BinImageReader(String filename) {
        this.filename = filename;
        Read();
    }

    private void Read() {
        try {
            InputStream is = getClass().getResourceAsStream(filename);
            BufferedInputStream bis = new BufferedInputStream(is);
            DataInputStream dis = new DataInputStream(bis);
            int magicNumber = ReadInt(bis);
            int nrOfImages = ReadInt(bis);
            int rows = ReadInt(bis);
            int columns = ReadInt(bis);

            int imageSize = rows * columns;
            result = new fmatrix(nrOfImages, rows * columns);

            byte[] pixelBlock = new byte[imageSize];
            for (int i = 0; i < nrOfImages; ++i) {
                dis.read(pixelBlock);
                for (int pixel = 0; pixel < imageSize; ++pixel) {
                    int cp = pixelBlock[pixel]&0xff; // &0xff;
                    float pixelValue = (float) ( cp / 255.0f)  ; // / 255.0f);
                    result.set(i, pixel, pixelValue);
                    /*float test = m_Result->get_value(i,pixel);
                        if ( test != pixelValue )
                                cout << "pixel misplaced\n";*/
                }
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
