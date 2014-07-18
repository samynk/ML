/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

/**
 *
 * @author Koen
 */
public class SingletonMemberShip extends MemberShip implements Cloneable{
    private float value;
    public SingletonMemberShip(String name,float value){
        super(name);
        this.value = value;
    }
    
    public float getValue(){
        return value;
    }
    
    public float getMinimumX(){
        return value;
    }
    
    public float getMaximumX(){
        return value;
    }
    
     public float evaluate(float x){
        if ( Math.abs(x - value) < 0.01f  )
            return 1.0f;
        else
            return 0.0f;
    }
     
     public void move(float dx) {
        value+=dx;
    }
     
     public String toString(){
         return Float.toString(value);
     }
     
    @Override
    public Object clone() throws CloneNotSupportedException {
        return new SingletonMemberShip(getName(),this.value);
    }
}
