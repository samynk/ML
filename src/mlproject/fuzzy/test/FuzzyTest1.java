/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy.test;

import mlproject.fuzzy.FuzzySystem;
import mlproject.fuzzy.FuzzyVariable;
import mlproject.fuzzy.LeftSigmoidMemberShip;
import mlproject.fuzzy.RightSigmoidMemberShip;
import mlproject.fuzzy.SigmoidMemberShip;
import mlproject.fuzzy.SingletonMemberShip;

/**
 *
 * @author Koen
 */
public class FuzzyTest1 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        FuzzySystem angleController = new FuzzySystem("armController");
        
        FuzzyVariable angle = new FuzzyVariable("angle");
        angle.addMemberShip(new LeftSigmoidMemberShip(-15,-5,"FARLEFT") );
        angle.addMemberShip(new SigmoidMemberShip(-15.0f,-5,0,"LEFT"));
        angle.addMemberShip(new SigmoidMemberShip(-5,0,5,"CENTER"));
        angle.addMemberShip(new SigmoidMemberShip(0,5,15,"RIGHT"));
        angle.addMemberShip(new RightSigmoidMemberShip(5,15,"FARRIGHT"));
        
        angleController.addFuzzyInput(angle);
        
        FuzzyVariable dAngle = new FuzzyVariable("dAngle");
        dAngle.addMemberShip(new SingletonMemberShip("STAY",0.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNLEFTSLOW",-2.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNLEFTFAST",-4.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNRIGHTSLOW",2.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNRIGHTFAST",4.0f));
        
        angleController.addFuzzyOutput(dAngle);
        
        angleController.addFuzzyRule("if angle is farleft then dAngle is turnrightfast");
        angleController.addFuzzyRule("if angle is left then dAngle is turnrightslow");
        angleController.addFuzzyRule("if angle is center then dAngle is stay");
        angleController.addFuzzyRule("if angle is right then dAngle is turnleftslow");
        angleController.addFuzzyRule("if angle is farright then dAngle is turnleftfast");
        
        angle.setCurrentValue(6.5f);
        
        angleController.evaluate();
        
        float result = angleController.getFuzzyOutput("dAngle");
        System.out.println("The result of the angle controller is : "+result);
    }
}
