/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * FuzzyVariableGUI.java
 *
 * Created on Nov 23, 2011, 4:41:44 PM
 */
package mlproject.fuzzy.gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Shape;
import java.awt.TexturePaint;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import mlproject.fuzzy.FuzzyVariable;
import mlproject.fuzzy.MemberShip;
import org.w3c.dom.DOMImplementation;
import org.w3c.dom.Document;

/**
 *
 * @author Koen
 */
public class FuzzyVariableGUI extends javax.swing.JPanel implements MouseListener, MouseMotionListener, ChangeListener {

    private float xmin = -10.0f;
    private float xmax = +10.0f;
    private BasicStroke basicStroke;
    private GeneralPath gp = new GeneralPath();
    private FuzzyVariable variable;
    private Color labelBackground = new Color(255, 255, 255, 200);
    private HashMap<String, Shape> memberShipShapes = new HashMap<>();
    private ArrayList<MemberShip> selectedMemberShips = new ArrayList<>();
    private DecimalFormat dec = new DecimalFormat("###.##");

    /**
     * Creates new form FuzzyVariableGUI
     */
    public FuzzyVariableGUI() {
        initComponents();
        basicStroke = new BasicStroke(0.1f);
        setMinimumSize(new Dimension(100, 100));
        setPreferredSize(new Dimension(100, 100));
        setMaximumSize(new Dimension(100, 100));
        setOpaque(true);
        addMouseListener(this);
        addMouseMotionListener(this);
        initHatch();

        valueRectangle = new RoundRectangle2D.Double();
    }

    public void setFuzzyVariable(FuzzyVariable variable) {
        if (this.variable != null) {
            variable.removeChangeListener(this);
        }

        //System.out.println("Variable " + variable.getName());
        xmin = variable.getMinimum();
        xmax = variable.getMaximum();
        
        // stretch the visible area a bit.
        float size = Math.abs(xmax - xmin)*0.025f;
        xmin -= size;
        xmax += size;
        
        this.variable = variable;
        if (this.variable != null) {
            variable.addChangeListener(this);
        }

        invalidate();
    }
    private BufferedImage hatchImage;
    private Rectangle2D hatchRectangle;
    private RoundRectangle2D valueRectangle;

    private void initHatch() {
        hatchImage = new BufferedImage(5, 5, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2 = hatchImage.createGraphics();
        g2.setColor(new Color(255, 255, 255, 127));
        g2.fillRect(0, 0, 5, 5);
        g2.setColor(new Color(24, 24, 255));
        g2.drawLine(0, 0, 5, 5); // \
        g2.drawLine(0, 5, 5, 0); // /

        hatchRectangle = new Rectangle2D.Double(0, 0, 5, 5);
    }
    @Override
    public void paintComponent(Graphics g){
         super.paintComponent(g);
        //System.out.println("painting component " + this.getWidth() + ","+ this.getHeight()+ " = " + variable);

        if (variable == null) {
            return;
        }
        paintVariable((Graphics2D)g);
    }

    public void paintVariable(Graphics2D g2d) {
        AffineTransform backup = g2d.getTransform();
        float scaleX = getWidth() / (xmax - xmin);
        float scaleY = -(getHeight() - 30);
        g2d.scale(scaleX, scaleY);
        g2d.translate(-xmin, -1);

        g2d.setStroke(basicStroke);

        int ixmax = (int) xmax;

        /*
         * for (int x = (int) xmin + 1; x <= ixmax; ++x) { g.drawLine(x, 0, x,
         * 1);
         *
         * }
         */

        memberShipShapes.clear();
        for (MemberShip ms : variable.getMemberShips()) {
            gp.reset();
            float starty = ms.evaluate(xmin);
            float lasty = 0.0f;
            gp.moveTo(xmin, starty);
            float dx = (xmax - xmin) / 200.0f;
            for (float x = xmin + dx; x < xmax; x += dx) {
                lasty = ms.evaluate(x);
                gp.lineTo(x, lasty);
            }
            if (lasty - starty > 1e-5) {
                gp.lineTo(xmax, starty);
            } else if (starty - lasty > 1e-5) {
                gp.lineTo(xmin, 0);
            }

            gp.closePath();
            g2d.setColor(ms.getColor());
            g2d.fill(gp);
            if (selectedMemberShips.contains(ms)) {
                hatchRectangle.setRect(0, 0, 5 / scaleX, 5 / scaleY);
                g2d.setPaint(new TexturePaint(hatchImage, hatchRectangle));
                g2d.fill(gp);
            }
            if (selectedMemberShips.contains(ms)) {
                g2d.setColor(Color.BLUE);
            } else {
                g2d.setColor(Color.BLACK);
            }
            g2d.draw(gp);

            memberShipShapes.put(ms.getName(), (GeneralPath) gp.clone());
        }

        // draw the current input value of the fuzzy variable
        float value = 0.0f;
        if (variable.isInput()) {
            value = variable.getInputValue();
        } else if (variable.isOutput()) {
            value = variable.getOutputValue();
        }

        float rvalue = value;
        if (rvalue > xmax) {
            rvalue = xmax;
        }
        if (rvalue < xmin) {
            rvalue = xmin;
        }

        g2d.setColor(new Color(255, 255, 24, 128));
        valueRectangle.setRoundRect(rvalue - 4 / scaleX, 0, 8 / scaleX, 1, (int) (8 / scaleX), (int) (8 / scaleY));
        g2d.fill(valueRectangle);
        g2d.setColor(Color.black);
        g2d.draw(valueRectangle);




        g2d.setTransform(backup);
        g2d.setColor(Color.gray);
        /*
         * for (int x = (int) xmin + 1; x <= ixmax; ++x) { float xpos =
         * ((x-xmin) * scaleX); g2d.drawString(Integer.toString(x), xpos,
         * getHeight());
        }
         */

        g2d.setColor(Color.black);
        FontMetrics fm = g2d.getFontMetrics();
        for (MemberShip ms : variable.getMemberShips()) {
            float minimum = ms.getMinimumX();
            float maximum = ms.getMaximumX();
            float centerPos = minimum + (maximum - minimum) / 2;
            float labelPos = (centerPos - xmin) * scaleX;

            String msLabel = ms.getName();
            int msLabelWidth = fm.stringWidth(msLabel);
            int iLabelPos = (int) (labelPos - msLabelWidth / 2);
            if (iLabelPos < 0) {
                iLabelPos = 0;
            }
            if (iLabelPos + msLabelWidth > getWidth()) {
                iLabelPos = getWidth() - msLabelWidth;
            }


            g2d.setPaint(labelBackground);
            g2d.fillRoundRect(iLabelPos - 2, getHeight() - 30 - fm.getAscent(), msLabelWidth + 4, fm.getMaxAscent() + fm.getMaxDescent(), 10, 10);
            g2d.setColor(Color.black);
            g2d.drawRoundRect(iLabelPos - 2, getHeight() - 30 - fm.getAscent(), msLabelWidth + 4, fm.getMaxAscent() + fm.getMaxDescent(), 10, 10);
            g2d.drawString(msLabel, iLabelPos, getHeight() - 30);

        }

        String sValue = dec.format(value);
        int swidth = fm.stringWidth(sValue) / 2;
        float labelX = (value - xmin) * scaleX;
        if (labelX - swidth < 0) {
            labelX = swidth * 2;
        } else if (labelX + swidth > getWidth()) {
            labelX = getWidth() - 2 * swidth - 10;
        } else {
            labelX -= swidth * 2;
        }

        int halfHeight = getHeight() / 5;
        g2d.setPaint(labelBackground);
        g2d.fillRoundRect((int) (labelX), halfHeight - fm.getAscent(), (int) (2 * swidth + 4), fm.getMaxAscent() + fm.getMaxDescent(), 10, 10);
        g2d.setColor(Color.black);
        g2d.drawRoundRect((int) (labelX), halfHeight - fm.getAscent(), (int) (2 * swidth + 4), fm.getMaxAscent() + fm.getMaxDescent(), 10, 10);
        g2d.drawString(sValue, labelX, halfHeight);

    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jButton2 = new javax.swing.JButton();

        setMinimumSize(new java.awt.Dimension(420, 150));
        setPreferredSize(new java.awt.Dimension(420, 150));
        addComponentListener(new java.awt.event.ComponentAdapter() {
            public void componentResized(java.awt.event.ComponentEvent evt) {
                formComponentResized(evt);
            }
        });

        jButton2.setIcon(new javax.swing.ImageIcon(getClass().getResource("/dae/images/Save.png"))); // NOI18N
        jButton2.setMargin(new java.awt.Insets(0, 0, 0, 0));
        jButton2.setMaximumSize(new java.awt.Dimension(24, 24));
        jButton2.setMinimumSize(new java.awt.Dimension(24, 24));
        jButton2.setPreferredSize(new java.awt.Dimension(24, 24));
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addGap(0, 396, Short.MAX_VALUE)
                .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addGap(0, 126, Short.MAX_VALUE)
                .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void formComponentResized(java.awt.event.ComponentEvent evt) {//GEN-FIRST:event_formComponentResized
        // TODO add your handling code here:
        System.out.println("resize event : " + getWidth() + "," + getHeight());
        float scaleX = getWidth() / (xmax - xmin);
        float scaleY = getHeight();

        float strokeScale = scaleX > scaleY ? scaleX : scaleY;
        basicStroke = new BasicStroke(1 / strokeScale);
        repaint();
    }//GEN-LAST:event_formComponentResized

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        /*
        Writer out = null;
        try {
            // TODO add your handling code here:
            // Get a DOMImplementation.
            DOMImplementation domImpl =
                    GenericDOMImplementation.getDOMImplementation();
            // Create an instance of org.w3c.dom.Document.
            String svgNS = "http://www.w3.org/2000/svg";
            Document document = domImpl.createDocument(svgNS, "svg", null);
            // Create an instance of the SVG Generator. 
            SVGGraphics2D svgGenerator = new SVGGraphics2D(document);
            this.paintVariable(svgGenerator);
            
            // Finally, stream out SVG to the standard output using
            // UTF-8 encoding.
            boolean useCSS = true; // we want to use CSS style attributes
            String currentDir = System.getProperty("user.dir");
            File saveDirectory = new File(currentDir,"memberships");
            if ( !saveDirectory.exists())
                saveDirectory.mkdir();
            File fileName = new File(saveDirectory, this.variable.getName() + ".svg");
            
            out = new OutputStreamWriter(new FileOutputStream(fileName), "UTF-8");
            svgGenerator.stream(out, useCSS);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(FuzzyVariableGUI.class.getName()).log(Level.SEVERE, null, ex);
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(FuzzyVariableGUI.class.getName()).log(Level.SEVERE, null, ex);
        } catch (SVGGraphics2DIOException ex) {
            Logger.getLogger(FuzzyVariableGUI.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                out.close();
            } catch (IOException ex) {
                Logger.getLogger(FuzzyVariableGUI.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        */
    }//GEN-LAST:event_jButton2ActionPerformed
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton2;
    // End of variables declaration//GEN-END:variables

    @Override
    public void mouseClicked(MouseEvent e) {
        if (!e.isControlDown()) {
            selectedMemberShips.clear();
        }

        float scaleX = getWidth() / (xmax - xmin);
        float scaleY = -(getHeight() - 10);
        float xpos = ((e.getX()) / scaleX) + xmin;
        float ypos = (e.getY() / scaleY) + 1;
        System.out.println("xpos, ypos :[" + xpos + "," + ypos + "]");
        for (String memberShip : memberShipShapes.keySet()) {
            Shape s = memberShipShapes.get(memberShip);
            // translate x and y to the shape coordinate system.

            if (s.contains(xpos, ypos)) {
                System.out.println("membership selected : " + memberShip);
                MemberShip ms = variable.getMemberShip(memberShip);
                if (e.isControlDown()) {
                    if (selectedMemberShips.contains(ms)) {
                        selectedMemberShips.remove(ms);
                    } else {
                        selectedMemberShips.add(ms);
                    }
                } else {
                    selectedMemberShips.add(ms);
                }
            }
        }
        repaint();
    }
    private Point2D.Float lastMousePress;
    private boolean startDraggingPossible = false;

    @Override
    public void mousePressed(MouseEvent e) {

        startDraggingPossible = false;
        Point2D.Float pos = convertToShapeCoordinates(e.getPoint());
        lastMousePress = pos;
        for (String memberShip : memberShipShapes.keySet()) {
            Shape s = memberShipShapes.get(memberShip);
            MemberShip ms = variable.getMemberShip(memberShip);
            if (!selectedMemberShips.contains(ms)) {
                continue;
            }
            if (s.contains(pos.x, pos.y)) {
                // start dragging possible.
                startDraggingPossible = true;
            }
        }
    }

    public Point2D.Float convertToShapeCoordinates(Point p) {
        float scaleX = getWidth() / (xmax - xmin);
        float scaleY = -(getHeight() - 10);
        float xpos = (p.x / scaleX) + xmin;
        float ypos = (p.y / scaleY) + 1;
        return new Point2D.Float(xpos, ypos);
    }

    @Override
    public void mouseReleased(MouseEvent e) {
    }

    @Override
    public void mouseEntered(MouseEvent e) {
    }

    @Override
    public void mouseExited(MouseEvent e) {
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        if (startDraggingPossible) {
            Point2D.Float pos = convertToShapeCoordinates(e.getPoint());
            float dx = pos.x - lastMousePress.x;
            //System.out.println("moving memberships : "+dx);
            for (MemberShip ms : selectedMemberShips) {
                ms.move(dx);
            }
            lastMousePress = pos;
            repaint();
        }
    }

    @Override
    public void mouseMoved(MouseEvent e) {
    }

    @Override
    public void stateChanged(ChangeEvent e) {
        /*
         * if (this.variable != null){ String text ="<html><body>";
         * for(MemberShip ms : variable.getMemberShips()){ float msValue =
         * ms.evaluate(variable.getInputValue());
         *
         * text += "<b>"+ms.getName() +"</b>:"+dec.format(msValue); text
         * +="<br>"; } text +="</body></html>"; lblLegend.setText(text);
         * lblLegend.invalidate(); this.doLayout();
        }
         */

        if (isVisible()) {
            repaint();
        }
    }

   
}
