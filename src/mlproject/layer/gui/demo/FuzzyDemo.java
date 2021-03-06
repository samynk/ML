/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer.gui.demo;

import java.io.File;
import java.io.FileFilter;
import mlproject.animation.Bone;

/**
 *
 * @author samyn_000
 */
public class FuzzyDemo extends javax.swing.JFrame {

    /**
     * Creates new form FuzzyDemo
     */
    public FuzzyDemo() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        fuzzySystemUI1 = new mlproject.fuzzy.gui.FuzzySystemUI();
        demoAnimationPanel1 = new mlproject.layer.gui.demo.DemoAnimationPanel();
        btnStep = new javax.swing.JButton();
        bodyRotationGraphPanel1 = new mlproject.fuzzy.gui.BodyRotationGraphPanel();
        jButton1 = new javax.swing.JButton();
        btnRecord = new javax.swing.JButton();
        btnStop = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Animation Synthesis");

        demoAnimationPanel1.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                demoAnimationPanel1ItemStateChanged(evt);
            }
        });

        javax.swing.GroupLayout demoAnimationPanel1Layout = new javax.swing.GroupLayout(demoAnimationPanel1);
        demoAnimationPanel1.setLayout(demoAnimationPanel1Layout);
        demoAnimationPanel1Layout.setHorizontalGroup(
            demoAnimationPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        demoAnimationPanel1Layout.setVerticalGroup(
            demoAnimationPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 548, Short.MAX_VALUE)
        );

        btnStep.setText("Step");
        btnStep.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnStepActionPerformed(evt);
            }
        });

        jButton1.setText("Play");
        jButton1.setActionCommand("btnPlay");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        btnRecord.setText("Record");
        btnRecord.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnRecordActionPerformed(evt);
            }
        });

        btnStop.setText("Stop");
        btnStop.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnStopActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(20, 20, 20)
                .addComponent(fuzzySystemUI1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(bodyRotationGraphPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, 480, Short.MAX_VALUE)
                    .addComponent(demoAnimationPanel1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addComponent(btnRecord)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(btnStop)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(jButton1)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(btnStep)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap(17, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(demoAnimationPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(bodyRotationGraphPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(btnStep)
                            .addComponent(jButton1)
                            .addComponent(btnRecord)
                            .addComponent(btnStop)))
                    .addComponent(fuzzySystemUI1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void demoAnimationPanel1ItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_demoAnimationPanel1ItemStateChanged
        // TODO add your handling code here:
        System.out.println("selected a bone");
        Bone b = (Bone) evt.getItem();
        this.fuzzySystemUI1.setFuzzySystem(b.getController());
        fuzzySystemUI1.repaint();

        bodyRotationGraphPanel1.addBone(b);
    }//GEN-LAST:event_demoAnimationPanel1ItemStateChanged

    private void btnStepActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnStepActionPerformed
        // TODO add your handling code here:
        demoAnimationPanel1.stepOne();
    }//GEN-LAST:event_btnStepActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        // TODO add your handling code here:
        demoAnimationPanel1.play();
    }//GEN-LAST:event_jButton1ActionPerformed

    private void btnRecordActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnRecordActionPerformed
        // TODO add your handling code here:
        String userdir = System.getProperty("user.dir");
        userdir += "/data";

        File directory = new File(userdir);
        if (!directory.exists()) {
            directory.mkdirs();
        }

        File[] files = directory.listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                if (pathname.isFile()) {
                    String path = pathname.getPath().toLowerCase();
                    return path.endsWith(".dat");
                } else {
                    return false;
                }
            }
        });
        int maxnumber = 0;

        String index = new String();
        for (File file : files) {
            String path = file.getName();
            int dotindex = path.indexOf(".");
            String name = path.substring(0, dotindex);


            for (int i = name.length() - 1; i > 0; --i) {
                if (Character.isDigit(name.charAt(i))) {
                    index = name.charAt(i) + index;
                } else {
                    break;
                }
            }



        }
        int fileIndex = 0;
        if (index.length() > 0) {
            fileIndex = Integer.parseInt(index) + 1;
        }

        demoAnimationPanel1.startRecording(new File(directory,"recording"+fileIndex+".dat"));
    }//GEN-LAST:event_btnRecordActionPerformed

    private void btnStopActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnStopActionPerformed
        // TODO add your handling code here:
        demoAnimationPanel1.stopRecording();
    }//GEN-LAST:event_btnStopActionPerformed

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
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(FuzzyDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(FuzzyDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(FuzzyDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(FuzzyDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new FuzzyDemo().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private mlproject.fuzzy.gui.BodyRotationGraphPanel bodyRotationGraphPanel1;
    private javax.swing.JButton btnRecord;
    private javax.swing.JButton btnStep;
    private javax.swing.JButton btnStop;
    private mlproject.layer.gui.demo.DemoAnimationPanel demoAnimationPanel1;
    private mlproject.fuzzy.gui.FuzzySystemUI fuzzySystemUI1;
    private javax.swing.JButton jButton1;
    // End of variables declaration//GEN-END:variables
}
