package dae.matrix;

/**
 * 
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class Intersection {
    public float3 intersection;
    public float t;

    public Intersection(float3 intersection, float t){
        this.intersection = intersection;
        this.t = t;
    }

    public boolean IsInside(){
        return t > 0 && t < 1;
    }

    public String toString(){
        return "Intersection : " + intersection.toString() + " , t = " + t;
    }
}
