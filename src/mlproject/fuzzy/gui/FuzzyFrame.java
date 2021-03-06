/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * FuzzyFrame.java
 *
 * Created on Nov 23, 2011, 4:47:40 PM
 */
package mlproject.fuzzy.gui;

import mlproject.fuzzy.FuzzySystem;

/**
 *
 * @author Koen
 */
public class FuzzyFrame extends javax.swing.JFrame {
    private FuzzySystem system;
    
    /** Creates new form FuzzyFrame */
    public FuzzyFrame() {
        initComponents();
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        fuzzySystemUI1 = new mlproject.fuzzy.gui.FuzzySystemUI();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Fuzzy Editor");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(fuzzySystemUI1, javax.swing.GroupLayout.DEFAULT_SIZE, 424, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(fuzzySystemUI1, javax.swing.GroupLayout.DEFAULT_SIZE, 313, Short.MAX_VALUE)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

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
            java.util.logging.Logger.getLogger(FuzzyFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {

            public void run() {
                new FuzzyFrame().setVisible(true);
            }
        });
    }
    
    public void setFuzzySystem(FuzzySystem system){
        this.system = system;
        fuzzySystemUI1.setFuzzySystem(system);
    }
    
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private mlproject.fuzzy.gui.FuzzySystemUI fuzzySystemUI1;
    // End of variables declaration//GEN-END:variables
}
