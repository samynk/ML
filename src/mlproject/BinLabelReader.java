/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject;

import intersect.data.fmatrix;
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
    
    private byte[] intBlock= new byte[4];
    public BinLabelReader(String filename){
        this.filename = filename;
        Read();
    }
    
    private void Read(){
        try {
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(filename));
            DataInputStream dis = new DataInputStream(bis);
            int magicNumber = ReadInt(bis);
            int nrOfItems = ReadInt(bis);
            
            result = new fmatrix(1,nrOfItems);
		
            byte[] pixelBlock = new byte[nrOfItems];
            dis.read(pixelBlock);
            for (int i = 1; i <= nrOfItems; ++i )
            {
                float label = (pixelBlock[i-1]&0xff);
                result.set(1, i, label);
            }
            dis.close();
        } catch (IOException ex) {
            Logger.getLogger(BinImageReader.class.getName()).log(Level.SEVERE, null, ex);
        } 
       
    }
    
    private int ReadInt(BufferedInputStream bis) throws IOException{
        bis.read(intBlock);
        int value = ((intBlock[0]&0xff) << 24);
        value += ((intBlock[1]&0xff)<<16);
        value += ((intBlock[2]&0xff)<<8);
        value += (intBlock[3]&0xff);
        return value;
    }
    
    public fmatrix getResult(){
        return result;
    }
}
