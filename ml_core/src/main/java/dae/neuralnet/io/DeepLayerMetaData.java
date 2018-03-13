/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.io;

import java.nio.file.Path;
import java.util.Calendar;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DeepLayerMetaData {

    private Calendar creationTime;
    private String author;
    private float succesRate;
    private Path path;

    public DeepLayerMetaData() {

    }

    /**
     * @return the creationTime
     */
    public Calendar getCreationTime() {
        return creationTime;
    }

    /**
     * @param creationTime the creationTime to set
     */
    public void setCreationTime(Calendar creationTime) {
        this.creationTime = creationTime;
    }

    /**
     * @return the author
     */
    public String getAuthor() {
        return author;
    }

    /**
     * @param author the author to set
     */
    public void setAuthor(String author) {
        this.author = author;
    }

    /**
     * @return the succesRate
     */
    public float getSuccesRate() {
        return succesRate;
    }

    /**
     * @param succesRate the succesRate to set
     */
    public void setSuccesRate(float succesRate) {
        this.succesRate = succesRate;
    }
    
    public Path getPath(){
        return path;
    }
    
    public void setPath(Path path){
        this.path = path;
    }
}
