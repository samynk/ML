/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.animation;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;

/**
 *
 * @author Koen
 */
public class AnimationTarget {
     private Point2D.Double location;
     private Ellipse2D.Double target;
     
     private double vx;
     private double vy;
     
     private double startX;
     private double startY;
     
     private double startVX;
     private double startVY;
     
     private boolean gravity;
     
     
     public AnimationTarget(double x, double y, double vx, double vy){
         location = new Point2D.Double(x,y);
         target = new Ellipse2D.Double(x,y,30,30);
         
         startX = x;
         startY = y;
         
         this.vx = vx;
         this.vy = vy;
         
         this.startVX = vx;
         this.startVY = vy;
     }
     
     public void setUseGravity(boolean gravity){
         this.gravity = gravity;
     }
     
     public boolean getUseGravity(){
         return gravity;
     }
     
     public void reset(){
         location.setLocation(startX,startY);
         target.setFrame(location.x, location.y, 30,30);
         
         this.vx = startVX;
         this.vy = startVY;
     }
     
     public void update(){
         location.x += vx;
         location.y += vy;
         if ( gravity )
         {
            vy += 0.2f;
         }
         target.setFrame(location.x -15, location.y - 15, 30,30);
     }
     
     public void draw(Graphics2D g2d){
         g2d.setPaint(Color.ORANGE);
         g2d.fill(target);
         g2d.setColor(Color.BLACK);
         g2d.draw(target);
     }
     
     public Point2D.Double getLocation(){
         return location;
     }
}
