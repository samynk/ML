/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.animation;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;
import java.awt.geom.RoundRectangle2D;
import java.util.ArrayList;
import mlproject.fuzzy.FuzzySystem;
import mlproject.layer.MultiLayerNN;

/**
 * This class describes a 2D bone object.
 *
 * @author Koen
 */
public class Bone {

    private Bone parent;
    private String name;
    private Point2D.Double location;
    /**
     * The length of this bone.
     */
    private double length;
    /**
     * The z rotation of this bone.
     */
    private double zRot;
    /**
     * The start rotation of this bone.
     */
    private double zStartRot;
    /**
     * The minimum rotation of the bone
     */
    private double rotMinimum;
    /**
     * The maximum rotation of the bone.
     */
    private double rotMaximum;
    /**
     * Stores the endpoint of this bone in worldspace
     */
    private Point2D.Double endPointWorld;
    /**
     * Stores the endpoint of this bone in localspace
     */
    private Point2D.Double endPointLocal;
    /**
     * the shape to draw for the bone.
     */
    private RoundRectangle2D.Double shape;
    private Ellipse2D.Double childDebugShape;
    /**
     * The affine transform for drawing the shape.
     */
    private AffineTransform boneTransform;
    /**
     * A helper object for the translation.
     */
    private AffineTransform boneRotationHelper;
    /**
     * The affine transform for drawing child elements.
     */
    private AffineTransform childTransform;
    private AffineTransform worldTransform;
    /**
     * The children of this bone.
     */
    private ArrayList<Bone> children = new ArrayList<>();
    private Controller controller;
    
    float time = 0.0f;
    /**
     * Creates a new Bone object with the given name.
     *
     * @param name
     */
    public Bone(String name, Point2D.Double location, double length, double zRotation, Controller controller, double rotMinimum, double rotMaximum) {
        this.name = name;
        this.location = location;
        this.length = length;
        this.zRot = zRotation * Math.PI / 180.0;
        if (zRot < rotMinimum) {
            zRot = rotMinimum;
        } else if (zRot > rotMaximum) {
            zRot = rotMaximum;
        }
        this.zStartRot = zRot;

        shape = new RoundRectangle2D.Double(0, -length / 10, length, length / 5, 10, 10);

        boneTransform = new AffineTransform();
        childTransform = new AffineTransform();
        boneRotationHelper = new AffineTransform();
        worldTransform = new AffineTransform();

        childDebugShape = new Ellipse2D.Double(-5, -5, 10, 10);
        this.controller = controller;

        this.rotMaximum = rotMaximum * Math.PI / 180;
        this.rotMinimum = rotMinimum * Math.PI / 180;

        endPointLocal = new Point2D.Double();
        endPointWorld = new Point2D.Double();
    }
    
    public Bone(String name, Point2D.Double location, double length, double zRotation, String controllerFile, double rotMinimum, double rotMaximum) {
        this.name = name;
        this.location = location;
        this.length = length;
        this.zRot = zRotation * Math.PI / 180.0;
        if (zRot < rotMinimum) {
            zRot = rotMinimum;
        } else if (zRot > rotMaximum) {
            zRot = rotMaximum;
        }
        this.zStartRot = zRot;

        shape = new RoundRectangle2D.Double(0, -length / 10, length, length / 5, 10, 10);

        boneTransform = new AffineTransform();
        childTransform = new AffineTransform();
        boneRotationHelper = new AffineTransform();
        worldTransform = new AffineTransform();

        childDebugShape = new Ellipse2D.Double(-5, -5, 10, 10);
        this.controller = new FuzzyController(controllerFile);

        this.rotMaximum = rotMaximum * Math.PI / 180;
        this.rotMinimum = rotMinimum * Math.PI / 180;

        endPointLocal = new Point2D.Double();
        endPointWorld = new Point2D.Double();
        
    
    }


    /**
     *
     * @param parent
     */
    public Bone(String name, Point2D.Double location, double width, double length, double zRotation,Controller controller , double rotMinimum, double rotMaximum) {
        this.name = name;
        this.location = location;
        this.length = length;
        this.zRot = zRotation * Math.PI / 180.0;
        if (zRot < rotMinimum) {
            zRot = rotMinimum;
        } else if (zRot > rotMaximum) {
            zRot = rotMaximum;
        }
        this.zStartRot = zRot;

        shape = new RoundRectangle2D.Double(0, -width / 2, length, width, 10, 10);

        boneTransform = new AffineTransform();
        childTransform = new AffineTransform();
        boneRotationHelper = new AffineTransform();
        worldTransform = new AffineTransform();

        childDebugShape = new Ellipse2D.Double(-5, -5, 10, 10);

        this.controller = controller;
        this.rotMaximum = rotMaximum * Math.PI / 180;
        this.rotMinimum = rotMinimum * Math.PI / 180;;

    }
    
      public Bone(String name, Point2D.Double location, double width, double length, double zRotation,String controllerFile , double rotMinimum, double rotMaximum) {
        this.name = name;
        this.location = location;
        this.length = length;
        this.zRot = zRotation * Math.PI / 180.0;
        if (zRot < rotMinimum) {
            zRot = rotMinimum;
        } else if (zRot > rotMaximum) {
            zRot = rotMaximum;
        }
        this.zStartRot = zRot;

        shape = new RoundRectangle2D.Double(0, -width / 2, length, width, 10, 10);

        boneTransform = new AffineTransform();
        childTransform = new AffineTransform();
        boneRotationHelper = new AffineTransform();
        worldTransform = new AffineTransform();

        childDebugShape = new Ellipse2D.Double(-5, -5, 10, 10);

        this.controller = new FuzzyController(controllerFile);
        this.rotMaximum = rotMaximum * Math.PI / 180;
        this.rotMinimum = rotMinimum * Math.PI / 180;;

        this.controller = new FuzzyController(controllerFile);
    }
    

    public void setParentBone(Bone parent) {
        this.parent = parent;
    }

//    public void setController(FuzzySystem controller) {
//        this.controller = controller;
//    }

    /**
     * Returns the parent bone of this bone.
     *
     * @return the parent bone.
     */
    public Bone getParentBone() {
        return parent;
    }

    /**
     * Adds a child to this bone.
     *
     * @param bone the child to add to the list of children.
     */
    public void addChild(Bone bone) {
        children.add(bone);
        bone.setParentBone(this);
    }

    /**
     * Returns the length of the bone.
     */
    public double getLength() {
        return length;
    }

    /**
     * Returns the rotation of the bone.
     */
    public double getRotation() {
        return zRot;
    }

    public void setRotation(double rotation) {
        this.zRot = rotation;
    }

    /**
     * Returns the name of the bone object.
     *
     * @return the name of the bone object.
     */
    public String getName() {
        return name;
    }

    /**
     * Gets the start transformation for children of this bone.
     *
     * @return the transformation for children of this bone.
     */
    public AffineTransform getChildTransform() {
        return childTransform;
    }

    public Point2D.Double getEndLocation() {
        childTransform.transform(endPointLocal, endPointWorld);
        return endPointWorld;
    }

    /**
     * Draws this bone on the screen.
     *
     * @param g the graphics to draw the bone with.
     */
    public void draw(Graphics2D g2d) {
        //AffineTransform backup = g2d.getTransform();
        //System.out.println("Drawing bone :" + zRot);
        g2d.setColor(Color.black);


        boneTransform.setToTranslation(location.x, location.y);
        boneRotationHelper.setToRotation(zRot);

        boneTransform.concatenate(boneRotationHelper);

        if (getParentBone() != null) {
            AffineTransform parentTransform = getParentBone().getChildTransform();
            boneTransform.preConcatenate(parentTransform);
        }

        Shape transformed = boneTransform.createTransformedShape(shape);
        g2d.setColor(Color.gray);
        g2d.fill(transformed);
        g2d.setColor(Color.black);
        g2d.draw(transformed);

        childTransform.setToTranslation(length, 0);
        childTransform.preConcatenate(boneTransform);

        Shape transformedDebug = boneTransform.createTransformedShape(childDebugShape);
        g2d.setColor(Color.blue);
        g2d.fill(transformedDebug);
        g2d.setColor(Color.black);
        g2d.draw(transformedDebug);

        for (Bone b : children) {
            b.draw(g2d);
        }
    }

    public void update(AnimationTarget target, Point2D.Double toMatch) {
        // calculate angle
        Point2D.Double newLoc = getWorldLocation();
        Point2D.Double tLoc = target.getLocation();

        double dx1 = tLoc.x - newLoc.x;
        double dy1 = tLoc.y - newLoc.y;

        double dx2 = toMatch.x - newLoc.x;
        double dy2 = toMatch.y - newLoc.y;

        float angle = (float) (Math.acos((dx1 * dx2 + dy1 * dy2) / (Math.hypot(dx1, dy1) * Math.hypot(dx2, dy2))) * 180.0 / Math.PI);

        double z = dx2 * dy1 - dx1 * dy2;
        if (z > 0) {
            angle = -angle;
        }


        System.out.println("angle : " + angle);

        //System.out.println("Angle is : " + angle);
        this.controller.setInput("angle", angle);

        double dx3 = tLoc.x - newLoc.x;
        double dy3 = tLoc.y - newLoc.y;
        float distance = (float) Math.hypot(dx3, dy3);
        float distance2 = (float) Math.hypot(dx2, dy2);


        //System.out.println("Distance is :" + (distance-distance2));

        // fuzzy controller

        this.controller.setInput("distance", (distance - distance2));

        controller.evaluate(time);
        time +=0.050f;
        //float dAngle = (this.controller.getFuzzyOutput("dAngle"))/50.0f;

        // neural controller


//        neuralController.setInput("distance", (distance - distance2));
//        neuralController.setInput("angle", angle);
//
//        neuralController.forward();
//        float dAngle = neuralController.getOutputLayer().getOutput(0) / 50.0f;
        
        float dAngle = controller.getOutput("dAngle")/50.0f;

        double newValue = 0.0f;
        if ( controller.isAbsoluteAngle()){
            newValue = zStartRot - dAngle;
        }else{
            newValue = zRot - dAngle;
        }
        if (newValue < this.rotMaximum && newValue > this.rotMinimum) {
            this.zRot = newValue;
        }
        System.out.println("Zrot is : " + zRot);

        addSample((float) (zRot * 180 / Math.PI));
    }
    private Point2D.Double origin = new Point2D.Double(0, 0);
    private Point2D.Double worldLoc = new Point2D.Double();
    private Point2D.Double relativeLocalLoc = new Point2D.Double();
    private Point2D.Double relativeWorldLoc = new Point2D.Double();

    public Point2D.Double getWorldLocation() {
        boneTransform.transform(origin, worldLoc);
        return worldLoc;
    }

    public Point2D.Double getWorldLocation(double x, double y) {
        relativeLocalLoc.setLocation(x, y);
        boneTransform.transform(relativeLocalLoc, relativeWorldLoc);
        return relativeWorldLoc;
    }

    public boolean isInside(double x, double y) {
        Shape transformed = boneTransform.createTransformedShape(shape);
        return transformed.contains(x, y);
    }

    public FuzzySystem getController() {
        if ( controller instanceof FuzzyController){
            return ((FuzzyController)controller).getFuzzySystem();
        }else
        return null;
    }

    public void reset() {
        this.zRot = zStartRot;
        time = 0.0f;
    }

    public MultiLayerNN getNeuralController() {
        return null;
    }
    public ArrayList<Float> samples = new ArrayList<>();
    private SampleSymbol sampleSymbol = SampleSymbol.SQUARE;

    public void addSample(float sample) {
        this.samples.add(sample);
    }

    public void resetSamples() {
        samples.clear();
    }

    public ArrayList<Float> getSamples() {
        return samples;
    }

    public SampleSymbol getSampleSymbol() {
        return sampleSymbol;
    }

    public float getTime() {
        return time;
    }
}