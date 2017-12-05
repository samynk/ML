/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * FuzzySystem.java
 *
 * Created on Dec 7, 2011, 1:04:08 PM
 */
package mlproject.fuzzy.gui;

import java.awt.GridBagConstraints;
import javax.swing.JLabel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
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
public class FuzzySystemUI extends javax.swing.JPanel implements ChangeListener {

    private FuzzySystem system;

    /**
     * Creates new form FuzzySystem
     */
    public FuzzySystemUI() {
        initComponents();

        FuzzySystem angleController = new FuzzySystem("armController",false);

        FuzzyVariable angle = new FuzzyVariable("angle");
        angle.addMemberShip(new LeftSigmoidMemberShip(-15, -5, "FARLEFT"));
        angle.addMemberShip(new SigmoidMemberShip(-15.0f, -5, 0, "LEFT"));
        angle.addMemberShip(new SigmoidMemberShip(-5, 0, 5, "CENTER"));
        angle.addMemberShip(new SigmoidMemberShip(0, 5, 15, "RIGHT"));
        angle.addMemberShip(new RightSigmoidMemberShip(5, 15, "FARRIGHT"));

        FuzzyVariable distance = new FuzzyVariable("distance");
        distance.addMemberShip(new LeftSigmoidMemberShip(0, 10, "NEAR"));
        distance.addMemberShip(new SigmoidMemberShip(0, 10, 30, "FAR"));
        distance.addMemberShip(new RightSigmoidMemberShip(10, 30, "VERYFAR"));

        angleController.addFuzzyInput(angle);
        angleController.addFuzzyInput(distance);

        FuzzyVariable dAngle = new FuzzyVariable("dAngle");
        dAngle.addMemberShip(new SingletonMemberShip("STAY", 0.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNLEFTSLOW", -2.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNLEFTFAST", -4.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNRIGHTSLOW", 2.0f));
        dAngle.addMemberShip(new SingletonMemberShip("TURNRIGHTFAST", 4.0f));

        angleController.addFuzzyOutput(dAngle);

        angleController.addFuzzyRule("if angle is farleft then dAngle is turnrightfast");
        angleController.addFuzzyRule("if angle is left then dAngle is turnrightslow");
        angleController.addFuzzyRule("if angle is center then dAngle is stay");
        angleController.addFuzzyRule("if angle is right then dAngle is turnleftslow");
        angleController.addFuzzyRule("if angle is farright then dAngle is turnleftfast");

        angle.setCurrentValue(6.5f);
        setFuzzySystem(angleController);
    }

    public final void setFuzzySystem(FuzzySystem system) {
        if (system == null) {
            return;
        }
        if (system == this.system) {
            return;
        }

        if (this.system != null) {
            this.system.removeChangeListener(this);
        }

        fuzzyInputs1.removeAll();
        fuzzyOutputs1.removeAll();


        this.system = system;

        if (this.system != null) {
            system.addChangeListener(this);
        }
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.weightx = 1.0f;
        gbc.weighty = 0.0f;
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.fill = GridBagConstraints.BOTH;
        for (FuzzyVariable input : this.system.getInputs()) {
            gbc.anchor = GridBagConstraints.WEST;
            JLabel lblVariableLabel = new JLabel();
            lblVariableLabel.setText(input.getName());
            fuzzyInputs1.add(lblVariableLabel, gbc);
            gbc.gridy++;
            FuzzyVariableGUI varUI = new FuzzyVariableGUI();
            varUI.setFuzzyVariable(input);
            fuzzyInputs1.add(varUI, gbc);
            gbc.gridy++;
        }

        gbc.weighty = 1.0f;
        fuzzyInputs1.add(new JLabel(), gbc);

        gbc = new GridBagConstraints();
        gbc.weightx = 1.0f;
        gbc.weighty = 0.0f;
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.fill = GridBagConstraints.BOTH;
        for (FuzzyVariable output : FuzzySystemUI.this.system.getOutputs()) {
            gbc.anchor = GridBagConstraints.WEST;
            JLabel lblVariableLabel = new JLabel();
            lblVariableLabel.setText(output.getName());
            fuzzyOutputs1.add(lblVariableLabel, gbc);
            gbc.gridy++;
            FuzzyVariableGUI varUI = new FuzzyVariableGUI();
            varUI.setFuzzyVariable(output);
            fuzzyOutputs1.add(varUI, gbc);
            gbc.gridy++;
        }
        gbc.weighty = 1.0f;
        fuzzyOutputs1.add(new JLabel(), gbc);
        updateRules();

        fuzzyInputs1.invalidate();
        fuzzyOutputs1.invalidate();

        fuzzyInputs1.revalidate();
        fuzzyOutputs1.revalidate();
        repaint();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {
        java.awt.GridBagConstraints gridBagConstraints;

        scrInputs = new javax.swing.JScrollPane();
        fuzzyInputs1 = new javax.swing.JPanel();
        scrOutputs = new javax.swing.JScrollPane();
        fuzzyOutputs1 = new javax.swing.JPanel();
        fuzzyRulePanel1 = new mlproject.fuzzy.gui.FuzzyRulePanel();

        setLayout(new java.awt.GridBagLayout());

        scrInputs.setBorder(javax.swing.BorderFactory.createTitledBorder("Fuzzy Inputs"));
        scrInputs.setMinimumSize(new java.awt.Dimension(100, 300));
        scrInputs.setPreferredSize(new java.awt.Dimension(100, 300));

        fuzzyInputs1.setLayout(new java.awt.GridBagLayout());
        scrInputs.setViewportView(fuzzyInputs1);

        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 0;
        gridBagConstraints.fill = java.awt.GridBagConstraints.BOTH;
        gridBagConstraints.weightx = 1.0;
        gridBagConstraints.weighty = 0.5;
        add(scrInputs, gridBagConstraints);

        scrOutputs.setBorder(javax.swing.BorderFactory.createTitledBorder("Fuzzy outputs"));
        scrOutputs.setMinimumSize(new java.awt.Dimension(100, 200));
        scrOutputs.setPreferredSize(new java.awt.Dimension(100, 300));

        fuzzyOutputs1.setMaximumSize(new java.awt.Dimension(200, 200));
        fuzzyOutputs1.setMinimumSize(new java.awt.Dimension(200, 200));
        fuzzyOutputs1.setLayout(new java.awt.GridBagLayout());
        scrOutputs.setViewportView(fuzzyOutputs1);

        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 1;
        gridBagConstraints.fill = java.awt.GridBagConstraints.BOTH;
        gridBagConstraints.weightx = 1.0;
        gridBagConstraints.weighty = 0.5;
        gridBagConstraints.insets = new java.awt.Insets(1, 1, 1, 1);
        add(scrOutputs, gridBagConstraints);

        fuzzyRulePanel1.setMinimumSize(new java.awt.Dimension(200, 200));

        javax.swing.GroupLayout fuzzyRulePanel1Layout = new javax.swing.GroupLayout(fuzzyRulePanel1);
        fuzzyRulePanel1.setLayout(fuzzyRulePanel1Layout);
        fuzzyRulePanel1Layout.setHorizontalGroup(
            fuzzyRulePanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 659, Short.MAX_VALUE)
        );
        fuzzyRulePanel1Layout.setVerticalGroup(
            fuzzyRulePanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 200, Short.MAX_VALUE)
        );

        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 2;
        gridBagConstraints.fill = java.awt.GridBagConstraints.BOTH;
        gridBagConstraints.weightx = 1.0;
        gridBagConstraints.weighty = 1.0;
        add(fuzzyRulePanel1, gridBagConstraints);
    }// </editor-fold>//GEN-END:initComponents

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JPanel fuzzyInputs1;
    private javax.swing.JPanel fuzzyOutputs1;
    private mlproject.fuzzy.gui.FuzzyRulePanel fuzzyRulePanel1;
    private javax.swing.JScrollPane scrInputs;
    private javax.swing.JScrollPane scrOutputs;
    // End of variables declaration//GEN-END:variables

    @Override
    public void stateChanged(ChangeEvent e) {
        updateRules();
    }

    private void updateRules() {
        //system.sortRules();
        fuzzyRulePanel1.setFuzzySystem(system);
    }

    public FuzzySystem getFuzzySystem() {
        return this.system;
    }
}
