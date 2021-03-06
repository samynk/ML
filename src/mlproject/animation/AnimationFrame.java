/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * AnimationFrame.java
 *
 * Created on Dec 6, 2011, 10:01:37 AM
 */
package mlproject.animation;

import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.FileFilter;

/**
 *
 * @author Koen
 */
public class AnimationFrame extends javax.swing.JFrame implements ItemListener{
    
    
    /** Creates new form AnimationFrame */
    public AnimationFrame() {
        initComponents();
        animationPanel1.addItemListener(this);
        animationPanel1.startAnimation();
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        btnRecord = new javax.swing.JToggleButton();
        btnFree = new javax.swing.JToggleButton();
        btnPlay = new javax.swing.JButton();
        btnPause = new javax.swing.JButton();
        btnStep = new javax.swing.JButton();
        btnReset = new javax.swing.JButton();
        animationPanel1 = new mlproject.animation.AnimationPanel();
        pnlOperations = new javax.swing.JPanel();
        pnlInfoPanel = new javax.swing.JPanel();
        pnlCurrentBone = new mlproject.animation.BonePanel();
        storedAnimationPanel1 = new mlproject.animation.StoredAnimationPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Machine Learning Project - Koen Samyn");
        setMinimumSize(new java.awt.Dimension(720, 480));

        jPanel1.setLayout(new java.awt.FlowLayout(java.awt.FlowLayout.RIGHT));

        btnRecord.setText("Record");
        btnRecord.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                btnRecordItemStateChanged(evt);
            }
        });
        jPanel1.add(btnRecord);

        btnFree.setText("Free");
        btnFree.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                btnFreeItemStateChanged(evt);
            }
        });
        jPanel1.add(btnFree);

        btnPlay.setText("Play");
        btnPlay.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnPlayActionPerformed(evt);
            }
        });
        jPanel1.add(btnPlay);

        btnPause.setText("Pause");
        btnPause.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnPauseActionPerformed(evt);
            }
        });
        jPanel1.add(btnPause);

        btnStep.setText("Step");
        btnStep.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnStepActionPerformed(evt);
            }
        });
        jPanel1.add(btnStep);

        btnReset.setText("Reset");
        btnReset.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnResetActionPerformed(evt);
            }
        });
        jPanel1.add(btnReset);

        getContentPane().add(jPanel1, java.awt.BorderLayout.SOUTH);

        javax.swing.GroupLayout animationPanel1Layout = new javax.swing.GroupLayout(animationPanel1);
        animationPanel1.setLayout(animationPanel1Layout);
        animationPanel1Layout.setHorizontalGroup(
            animationPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 660, Short.MAX_VALUE)
        );
        animationPanel1Layout.setVerticalGroup(
            animationPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 605, Short.MAX_VALUE)
        );

        getContentPane().add(animationPanel1, java.awt.BorderLayout.CENTER);

        pnlOperations.setMinimumSize(new java.awt.Dimension(240, 108));
        pnlOperations.setPreferredSize(new java.awt.Dimension(240, 623));
        pnlOperations.setLayout(new javax.swing.BoxLayout(pnlOperations, javax.swing.BoxLayout.PAGE_AXIS));

        pnlInfoPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Info"));
        pnlInfoPanel.setMinimumSize(new java.awt.Dimension(720, 480));
        pnlInfoPanel.setPreferredSize(new java.awt.Dimension(720, 480));

        javax.swing.GroupLayout pnlInfoPanelLayout = new javax.swing.GroupLayout(pnlInfoPanel);
        pnlInfoPanel.setLayout(pnlInfoPanelLayout);
        pnlInfoPanelLayout.setHorizontalGroup(
            pnlInfoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(pnlInfoPanelLayout.createSequentialGroup()
                .addComponent(pnlCurrentBone, javax.swing.GroupLayout.DEFAULT_SIZE, 175, Short.MAX_VALUE)
                .addGap(258, 258, 258))
        );
        pnlInfoPanelLayout.setVerticalGroup(
            pnlInfoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(pnlInfoPanelLayout.createSequentialGroup()
                .addComponent(pnlCurrentBone, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(256, Short.MAX_VALUE))
        );

        pnlOperations.add(pnlInfoPanel);

        storedAnimationPanel1.setBorder(javax.swing.BorderFactory.createTitledBorder("Machine learning"));
        storedAnimationPanel1.setMinimumSize(new java.awt.Dimension(200, 108));
        pnlOperations.add(storedAnimationPanel1);

        getContentPane().add(pnlOperations, java.awt.BorderLayout.WEST);

        java.awt.Dimension screenSize = java.awt.Toolkit.getDefaultToolkit().getScreenSize();
        setBounds((screenSize.width-918)/2, (screenSize.height-685)/2, 918, 685);
    }// </editor-fold>//GEN-END:initComponents

    private void btnPauseActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnPauseActionPerformed
        // TODO add your handling code here:
        animationPanel1.pause();
        transferBones();
    }//GEN-LAST:event_btnPauseActionPerformed

    private void btnPlayActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnPlayActionPerformed
        // TODO add your handling code here:
        animationPanel1.play();
        transferBones();
    }//GEN-LAST:event_btnPlayActionPerformed

    private void btnStepActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnStepActionPerformed
        // TODO add your handling code here:
        animationPanel1.stepOne();
        transferBones();
    }//GEN-LAST:event_btnStepActionPerformed

    private void btnFreeItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_btnFreeItemStateChanged
        // TODO add your handling code here:
        animationPanel1.toggleFree();
        transferBones();
    }//GEN-LAST:event_btnFreeItemStateChanged

    private void btnResetActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnResetActionPerformed
        // TODO add your handling code here:
        animationPanel1.reset();
        animationPanel1.pause();
        animationPanel1.repaint();
    }//GEN-LAST:event_btnResetActionPerformed

    private void transferBones(){
        storedAnimationPanel1.setHandBone(animationPanel1.getHandBone());
        storedAnimationPanel1.setUpperArmBone(animationPanel1.getUpperArmBone());
        storedAnimationPanel1.setLowerArmBone(animationPanel1.getLowerArmBone());
        
    }
    
    private void btnRecordItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_btnRecordItemStateChanged
        // TODO add your handling code here:
        if ( btnRecord.isSelected()){
            // start recording.
            String currentDir = System.getProperty("user.dir");
            File subdir = new File(currentDir,"motioncurves");
            
            if ( !subdir.exists())
                subdir.mkdir();
            
            File[] existing = subdir.listFiles(new FileFilter(){
                @Override
                public boolean accept(File pathname) {
                   return (pathname.getPath().toLowerCase().endsWith(".mo"));
                }
            });
            int maxNumber = 0;
            for ( File file: existing){
                String path = file.getPath();
                int startIndex = path.indexOf('_')+1;
                int endIndex = path.lastIndexOf('.');
                int number = Integer.parseInt(path.substring(startIndex,endIndex));
                if ( number > maxNumber){
                    maxNumber = number ;
                }
            }
            animationPanel1.startRecording(new File(subdir,"motion_"+(maxNumber+1)+".mo"));
        }else{
            animationPanel1.stopRecording();
        }
    }//GEN-LAST:event_btnRecordItemStateChanged

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(AnimationFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {

            public void run() {
                new AnimationFrame().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private mlproject.animation.AnimationPanel animationPanel1;
    private javax.swing.JToggleButton btnFree;
    private javax.swing.JButton btnPause;
    private javax.swing.JButton btnPlay;
    private javax.swing.JToggleButton btnRecord;
    private javax.swing.JButton btnReset;
    private javax.swing.JButton btnStep;
    private javax.swing.JPanel jPanel1;
    private mlproject.animation.BonePanel pnlCurrentBone;
    private javax.swing.JPanel pnlInfoPanel;
    private javax.swing.JPanel pnlOperations;
    private mlproject.animation.StoredAnimationPanel storedAnimationPanel1;
    // End of variables declaration//GEN-END:variables

    @Override
    public void itemStateChanged(ItemEvent e) {
        if (e.getStateChange() == ItemEvent.SELECTED)
        {
            Bone b = (Bone)e.getItem();
            this.pnlCurrentBone.setCurrentBone(b);
        }
    }
}
