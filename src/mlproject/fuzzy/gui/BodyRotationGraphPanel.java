/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy.gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.GeneralPath;
import java.io.*;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.Timer;
import mlproject.animation.Bone;
import mlproject.animation.SampleSymbol;
import org.w3c.dom.DOMImplementation;
import org.w3c.dom.Document;

/**
 *
 * @author Koen
 */
public class BodyRotationGraphPanel extends javax.swing.JPanel implements ActionListener {

    private GeneralPath gp = new GeneralPath();
    private static Color graphColors[];
    private int minX = 0;
    private int maxX = 100;
    
    private ArrayList<Bone> bones = new ArrayList<>();

    static {
        graphColors = new Color[8];
        graphColors[0] = new Color(255, 24, 24, 255);
        graphColors[1] = new Color(24, 255, 24, 255);
        graphColors[2] = new Color(227, 227, 227, 255);
        graphColors[3] = new Color(24, 24, 255, 255);
        graphColors[4] = new Color(255, 255, 255);
        graphColors[5] = new Color(255, 0, 0);
        graphColors[6] = new Color(0, 255, 0);
        graphColors[7] = new Color(0, 0, 255);
    }

    public void addBone(Bone b) {
        if ( !bones.contains(b))
            bones.add(b);
    }

    @Override
    public void actionPerformed(ActionEvent ae) {
        repaint();
    }

    /**
     * Creates new form BodyRotationGraphPanel
     */
    public BodyRotationGraphPanel() {
        initComponents();
         Timer t = new Timer(50, this);
        t.setDelay(100);
        t.start();
    }
    private BasicStroke graphStroke = new BasicStroke(1.0f);
    private BasicStroke gridStroke = new BasicStroke(1.0f);

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        //System.out.println("Sample size : " + samples.isEmpty());
        g.setColor(Color.green);
        float stepWidth = this.getWidth() * 1.0f / 100;
        ((Graphics2D) g).setStroke(gridStroke);

        for (int gx = 0; gx < this.getWidth(); gx += stepWidth * 10) {
            g.setColor(Color.lightGray);
            g.drawLine(gx, 0, gx, getHeight());
        }

        float yScale = this.getHeight() / 180.0f;

        for (int gy = -90; gy <= 90; gy += 10) {
            int height = (int) (-gy * yScale) + getHeight() / 2;
            g.setColor(Color.lightGray);
            g.drawLine(0, height, getWidth(), height);
            g.setColor(Color.white);
            g.drawString(Integer.toString(gy), 10, height);
        }

        g.setColor(Color.red);
        int pMinX = (this.minX * getWidth()) / 100;
        int pMaxX = (this.maxX * getWidth()) / 100;
        g.drawLine(pMinX, 0, pMinX, getHeight());
        g.drawLine(pMaxX, 0, pMaxX, getHeight());
        int ci = 0;
        for (Bone b: bones) {
            gp.reset();
            float currentX = this.getWidth();
            ArrayList<Float> sampleList = b.getSamples();
            SampleSymbol symbol = b.getSampleSymbol();
            if (sampleList.size() > 1) {
                gp.moveTo(currentX, -yScale * sampleList.get(sampleList.size() - 1) + this.getHeight() / 2);
                currentX -= stepWidth;
                for (int i = sampleList.size() - 2; i >= 0; --i) {
                    float currentY = -yScale * sampleList.get(i) + this.getHeight() / 2;
                    gp.lineTo(currentX, currentY);
                    currentX -= stepWidth;
                    if (i % 20 == 0) {
                        g.setColor(graphColors[ci % graphColors.length]);
                        int iCurrentX = (int) currentX;
                        int iCurrentY = (int) currentY;
                        switch (symbol) {
                            case SQUARE:
                                g.drawRect(iCurrentX - 3, iCurrentY - 3, 6, 6);
                                break;
                            case CIRCLE:
                                g.drawOval(iCurrentX - 3, iCurrentY - 3, 6, 6);
                                break;
                            case TRIANGLE:
                                g.drawLine(iCurrentX - 3, iCurrentY - 3, iCurrentX + 3, iCurrentY - 3);
                                g.drawLine(iCurrentX + 3, iCurrentY - 3, iCurrentX, iCurrentY + 3);
                                g.drawLine(iCurrentX, iCurrentY + 3, iCurrentX - 3, iCurrentY - 3);
                                break;
                        }
                    }
                }
            }
            g.setColor(graphColors[ci % graphColors.length]);
            ((Graphics2D) g).setStroke(graphStroke);
            ((Graphics2D) g).draw(gp);
            ++ci;
        }
    }

    public void paintVariable(Graphics2D g2d) {
       
        //System.out.println("Sample size : " + samples.isEmpty());
        g2d.setColor(Color.green);

        g2d.setStroke(gridStroke);


        int sbc = 100;
        int minIndex = sbc - (this.minX * sbc) / 100;
        int maxIndex = sbc - (this.maxX * sbc) / 100;
        int width = 300;
        float stepWidth = 300.0f / Math.abs(maxIndex - minIndex);


        float maxY = Float.NEGATIVE_INFINITY;
        float minY = Float.POSITIVE_INFINITY;

        for (Bone b : bones) {
            ArrayList<Float> sampleList = b.getSamples();
            for (float v : sampleList) {
                if (v > maxY) {
                    maxY = v;
                }
                if (v < minY) {
                    minY = v;
                }
            }
        }
        int iMinY = ((int) (minY / 10)) * 10 - 10;
        int iMaxY = ((int) (maxY / 10)) * 10 + 10;

        float yScale = (this.getHeight() * 1.0f) / Math.abs(iMaxY - iMinY);
        for (int gy = iMinY; gy <= iMaxY; gy += 10) {
            int height = (int) (getHeight() - yScale * (gy - iMinY));
            //System.out.println("Height for " + gy + " is : " + height);
            g2d.setColor(Color.lightGray);
            g2d.drawLine(0, height, width, height);
            g2d.setColor(Color.black);
            g2d.drawString(Integer.toString(gy), -30, height + g2d.getFontMetrics().getDescent());
        }

        int nrOfSecs = (int) (width / (stepWidth * 20));
        for (int sec = 1; sec <= nrOfSecs; sec++) {
            g2d.setColor(Color.lightGray);
            int lx = (int) (sec * stepWidth * 20);
            g2d.drawLine(lx, 0, lx, getHeight());
            // TODO : sample rate is 50 ms 
            g2d.setColor(Color.black);
            String label = Integer.toString(sec);
            g2d.drawString(label, lx - g2d.getFontMetrics().stringWidth(label) / 2, getHeight() + 20);
        }

        g2d.setColor(Color.lightGray);
        g2d.drawRect(0, 0, width, getHeight());

        int ci = 0;
        for (Bone b : bones) {
            gp.reset();

            ArrayList<Float> sampleList = b.getSamples();
            SampleSymbol symbol = b.getSampleSymbol();
            if (sampleList.size() > 1) {
                sbc = 100;
                minIndex = sbc - (this.minX * sbc) / 100;
                maxIndex = sbc - (this.maxX * sbc) / 100;
                if (maxIndex >= sampleList.size()) {
                    continue;
                } else {
                    maxIndex = sampleList.size() - maxIndex;
                }
                if (minIndex >= sampleList.size()) {
                    minIndex = 0;
                } else {
                    minIndex = sampleList.size() - minIndex;
                }

                float currentX = width;
                gp.moveTo(currentX, getHeight() - yScale * (sampleList.get(maxIndex) - iMinY));
                currentX -= stepWidth;
                for (int i = maxIndex - 1; i >= minIndex; --i) {
                    float currentY = getHeight() - yScale * (sampleList.get(i) - iMinY);
                    gp.lineTo(currentX, currentY);
                    currentX -= stepWidth;
                    if (i % 20 == 0) {
                        g2d.setColor(Color.black);
                        int iCurrentX = Math.round(currentX);
                        int iCurrentY = Math.round(currentY);
                        switch (symbol) {
                            case SQUARE:
                                g2d.drawRect(iCurrentX - 4, iCurrentY - 4, 9, 9);
                                break;
                            case CIRCLE:
                                g2d.drawOval(iCurrentX - 4, iCurrentY - 4, 9, 9);
                                break;
                            case TRIANGLE:
                                g2d.drawLine(iCurrentX - 4, iCurrentY - 4, iCurrentX + 4, iCurrentY + 4);
                                g2d.drawLine(iCurrentX + 4, iCurrentY - 4, iCurrentX - 4, iCurrentY + 4);
                                break;
                        }
                    }
                }
            }
            g2d.setColor(Color.black);
            g2d.setStroke(graphStroke);
            g2d.draw(gp);
            ++ci;
        }
        String xaxis = "time (s)";
        int xAxisWidth = g2d.getFontMetrics().stringWidth(xaxis);

        g2d.drawString(xaxis, width / 2 - xAxisWidth / 2, getHeight() + 40 + g2d.getFontMetrics().getAscent());



        // legend
        int y = 20;
        int x = 10;
        for (Bone b : bones) {
            
            SampleSymbol symbol = b.getSampleSymbol();
            switch (symbol) {
                case SQUARE:
                    g2d.drawRect(x - 4, y - 4, 8, 8);
                    break;
                case CIRCLE:
                    g2d.drawOval(x - 4, y - 4, 6, 8);
                    break;
                case TRIANGLE:
                    g2d.drawLine(x - 4, y - 4, x + 4, y + 4);
                    g2d.drawLine(x + 4, y - 4, x - 4, y + 4);
                    break;
            }
            g2d.drawString(b.getName(), x + 10, y + 4);
            y += g2d.getFontMetrics().getHeight() + 5;
        }

        g2d.rotate(-Math.PI / 2);
        String yaxis = "rotation (°)";
        int yAxisWidth = g2d.getFontMetrics().stringWidth(yaxis);
        g2d.drawString("rotation ($\\circ$)", -getHeight() / 2 - yAxisWidth / 2, -50);
    }

    public void recordGraph() {
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
            File saveDirectory = new File(currentDir, "graphs");
            if (!saveDirectory.exists()) {
                saveDirectory.mkdir();
            }
            File fileName = new File(saveDirectory, "graph.svg");

            out = new OutputStreamWriter(new FileOutputStream(fileName), "UTF-8");
            svgGenerator.stream(out, useCSS);
        } catch (FileNotFoundException | UnsupportedEncodingException ex) {
            Logger.getLogger(FuzzyVariableGUI.class.getName()).log(Level.SEVERE, null, ex);
        } catch (SVGGraphics2DIOException ex) {
        } finally {
            try {
                out.close();
            } catch (IOException ex) {
                Logger.getLogger(FuzzyVariableGUI.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        */
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        setPreferredSize(new java.awt.Dimension(400, 400));
        setLayout(new java.awt.FlowLayout(java.awt.FlowLayout.LEFT));
    }// </editor-fold>//GEN-END:initComponents
    // Variables declaration - do not modify//GEN-BEGIN:variables
    // End of variables declaration//GEN-END:variables

    /**
     * @return the minX
     */
    public int getMinX() {
        return minX;
    }

    /**
     * @param minX the minX to set
     */
    public void setMinX(int minX) {
        this.minX = minX;
    }

    /**
     * @return the maxX
     */
    public int getMaxX() {
        return maxX;
    }

    /**
     * @param maxX the maxX to set
     */
    public void setMaxX(int maxX) {
        this.maxX = maxX;
    }
}
