/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * BonePanel.java
 *
 * Created on Dec 7, 2011, 9:33:35 AM
 */
package mlproject.animation;

import java.awt.Frame;
import mlproject.fuzzy.gui.FuzzySystemDialog;
import mlproject.layer.gui.MultiLayerDialog;

/**
 *
 * @author Koen
 */
public class BonePanel extends javax.swing.JPanel {

    private Bone current;
    static FuzzySystemDialog fuzzyDialog;

    /** Creates new form BonePanel */
    public BonePanel() {
        initComponents();
    }

    public void setCurrentBone(Bone current) {
        if (current != null) {
            this.current = current;

            txtName.setText(current.getName());
            int rotation = (int) (current.getRotation() * 180.0 / Math.PI);
            spnRotation.setValue(rotation);
        }
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        lblName = new javax.swing.JLabel();
        txtName = new javax.swing.JTextField();
        lblRotation = new javax.swing.JLabel();
        spnRotation = new javax.swing.JSpinner();
        lblMinRotation = new javax.swing.JLabel();
        spnMinRotation = new javax.swing.JSpinner();
        lblMaximumRotation = new javax.swing.JLabel();
        spnMaxRotation = new javax.swing.JSpinner();
        btnEditController = new javax.swing.JButton();
        btnViewNeuralNet = new javax.swing.JButton();

        setMinimumSize(new java.awt.Dimension(220, 220));
        setPreferredSize(new java.awt.Dimension(220, 220));

        lblName.setText("Name : ");

        lblRotation.setText("Rotation : ");

        spnRotation.setModel(new javax.swing.SpinnerNumberModel(Integer.valueOf(0), null, Integer.valueOf(360), Integer.valueOf(1)));
        spnRotation.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                spnRotationStateChanged(evt);
            }
        });

        lblMinRotation.setText("Minimum rotation : ");

        lblMaximumRotation.setText("Maximum rotation :");

        btnEditController.setText(" Edit Controller ...");
        btnEditController.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnEditControllerActionPerformed(evt);
            }
        });

        btnViewNeuralNet.setText("View Neural Net ...");
        btnViewNeuralNet.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnViewNeuralNetActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(lblName)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(txtName, javax.swing.GroupLayout.DEFAULT_SIZE, 192, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(lblRotation)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(spnRotation, javax.swing.GroupLayout.DEFAULT_SIZE, 178, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(lblMinRotation)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(spnMinRotation, javax.swing.GroupLayout.DEFAULT_SIZE, 125, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(lblMaximumRotation)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(spnMaxRotation, javax.swing.GroupLayout.DEFAULT_SIZE, 126, Short.MAX_VALUE))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                        .addComponent(btnEditController, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(btnViewNeuralNet, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblName)
                    .addComponent(txtName, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblRotation)
                    .addComponent(spnRotation, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblMinRotation)
                    .addComponent(spnMinRotation, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblMaximumRotation)
                    .addComponent(spnMaxRotation, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnEditController)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(btnViewNeuralNet)
                .addContainerGap(34, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void spnRotationStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_spnRotationStateChanged
        int rotation = (Integer) spnRotation.getValue();
        double zRot = rotation * Math.PI / 180.0;
        current.setRotation(zRot);
    }//GEN-LAST:event_spnRotationStateChanged

    private void btnEditControllerActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnEditControllerActionPerformed
        // TODO add your handling code here:
        if (current != null) {
            if (fuzzyDialog == null) {
                fuzzyDialog = new FuzzySystemDialog((Frame) this.getTopLevelAncestor(), false);
            }
            fuzzyDialog.setFuzzySystem(this.current.getController());
            fuzzyDialog.setVisible(true);
        }
    }//GEN-LAST:event_btnEditControllerActionPerformed

    private void btnViewNeuralNetActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnViewNeuralNetActionPerformed
        // TODO add your handling code here:
        MultiLayerDialog mld = new MultiLayerDialog((Frame) this.getTopLevelAncestor(), false);
        mld.setMultiLayerNN(this.current.getNeuralController());
        mld.setVisible(true);
    }//GEN-LAST:event_btnViewNeuralNetActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnEditController;
    private javax.swing.JButton btnViewNeuralNet;
    private javax.swing.JLabel lblMaximumRotation;
    private javax.swing.JLabel lblMinRotation;
    private javax.swing.JLabel lblName;
    private javax.swing.JLabel lblRotation;
    private javax.swing.JSpinner spnMaxRotation;
    private javax.swing.JSpinner spnMinRotation;
    private javax.swing.JSpinner spnRotation;
    private javax.swing.JTextField txtName;
    // End of variables declaration//GEN-END:variables
}
