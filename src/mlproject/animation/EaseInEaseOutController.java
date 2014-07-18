/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.animation;

/**
 *
 * @author samyn_000
 */
public class EaseInEaseOutController implements Controller {
    // difference between the start orientation and target orientation
    float startAngle;
    float angle;
    float d;
    float change;
    
    float output;
    
    public EaseInEaseOutController(float duration,float totalChange,float startAngle) {
        this.d = duration;
        this.startAngle = startAngle;
        this.change = totalChange;
    }

    @Override
    public void evaluate(float time) {
        float t = time/ (d/2.0f);
        if (t < 1){
            output = (change/2)*t*t*t;
        }else if(t < 2){
            t -= 2;
            output =  (change/2)*(t*t*t + 2);
        }else{
            output = change;
        }
        output *= Math.PI / 180.0;
    }

    @Override
    public void setInput(String name, float value) {
        if ( "angle".equals(name)){
            this.angle = value;
        }
    }

    @Override
    public float getOutput(String name) {
        return output;
    }
    
    
    @Override
    public boolean isAbsoluteAngle(){
        return true;
    }
}
