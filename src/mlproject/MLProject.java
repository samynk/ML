/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject;

import intersect.data.fmatrix;

/**
 *
 * @author Koen
 */
public class MLProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        BinImageReader bir = new BinImageReader("./Data/train-images.idx3-ubyte.bin");
	fmatrix result = bir.getResult();
        
        // columns contain 28x28 images
        
    }
}
