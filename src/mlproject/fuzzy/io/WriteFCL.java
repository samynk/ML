/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy.io;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import mlproject.fuzzy.FuzzyRule;
import mlproject.fuzzy.FuzzySystem;
import mlproject.fuzzy.FuzzyVariable;
import mlproject.fuzzy.MemberShip;

/**
 * REMARK: not tested, treat as buggy
 * @author Koen
 */
public class WriteFCL {
    private String filename;
    
    public WriteFCL(String filename)
    {
        this.filename = filename;
    }
    
    public void write(FuzzySystem system)
    {
        FileWriter fw;
        try {
            fw = new FileWriter(filename);
            BufferedWriter bw = new BufferedWriter(fw);
            // inputs
            bw.write("FUNCTION BLOCK\n");
            bw.write("\nVAR_INPUT\n");
            for(FuzzyVariable input : system.getInputs())
            {
                bw.write(input.getName());
                bw.write(":");
                bw.write("REAL;\n");
                        
            }
            bw.write("END_VAR\n");
            
            for(FuzzyVariable input : system.getInputs())
            {
                bw.write("\nFUZZIFY ");
                bw.write(input.getName());
                bw.write("\n");
                
                for ( MemberShip ms : input.getMemberShips())
                {
                    bw.write("TERM ");
                    bw.write(ms.getName());
                    bw.write(" := ");
                    bw.write(ms.toString());
                    bw.write("\n");
                }
                
                bw.write("END_FUZZIFY\n");
            }
            
            // outputs
             bw.write("\nVAR_OUTPUT\n");
            for(FuzzyVariable output : system.getOutputs())
            {
                bw.write(output.getName());
                bw.write(":");
                bw.write("REAL;\n");
                        
            }
            bw.write("END_VAR\n");
            
            for(FuzzyVariable output : system.getOutputs())
            {
                bw.write("\nDEFUZZIFY ");
                bw.write(output.getName());
                bw.write("\n");
                
                for ( MemberShip ms : output.getMemberShips())
                {
                    bw.write("TERM ");
                    bw.write(ms.getName());
                    bw.write(" := ");
                    bw.write(ms.toString());
                    bw.write("\n");
                }
                
                bw.write("END_DEFUZZIFY\n");
            }
            
            // rules
            bw.write("\nRULEBLOCK\n");
            
            int i = 1;
            for ( FuzzyRule rule: system.getRules())
            {
                bw.write("RULE ");
                bw.write(Integer.toString(i));
                bw.write(":");
                bw.write(rule.getRuleText());
                bw.write("\n");
                ++i;
            }
            bw.write("END_RULEBLOCK\n");
            bw.write("END_FUNCTIONBLOCK\n");
            bw.close();
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
    }
}
