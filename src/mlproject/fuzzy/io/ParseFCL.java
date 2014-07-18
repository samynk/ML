/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy.io;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import mlproject.fuzzy.*;
import mlproject.fuzzy.gui.FuzzyFrame;

/**
 *
 * @author Koen
 */
public class ParseFCL {

    enum PARSESTATE {

        INPUTS, OUTPUTS, INPUTMEMBER, OUTPUTMEMBER, RULES, IDLE
    }
    private PARSESTATE state;
    private FuzzySystem result;
    private FuzzyVariable currentInput;
    private FuzzyVariable currentOutput;
    private int currentRuleGroup = 1;

    public ParseFCL(String filename) {
        try {
            FileReader fr = new FileReader(filename);
            BufferedReader br = new BufferedReader(fr);
            read(filename, br);
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
    }

    public ParseFCL(InputStream openStream) {
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(openStream));
            read("system", br);
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
    }

    private void read(String systemName, BufferedReader br) throws IOException {
        String line;
        result = new FuzzySystem(systemName);
        state = PARSESTATE.IDLE;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("#")) {
                System.out.println("comment : " + line.substring(1));
            } else if (line.startsWith("VAR_INPUT")) {
                state = PARSESTATE.INPUTS;
            } else if (line.startsWith("END_VAR")) {
                state = PARSESTATE.IDLE;
            } else if (line.startsWith("FUZZIFY")) {
                String[] cs = line.split("\\s+");
                if (cs.length == 2) {
                    //System.out.println("Getting variable : " + cs[1]);
                    currentInput = result.getFuzzyInputVariable(cs[1]);
                    if ( currentInput == null){
                        System.out.println("Could not find : " + cs[1]);
                    }
                }
                state = PARSESTATE.INPUTMEMBER;
            } else if (line.startsWith("VAR_OUTPUT")) {
                state = PARSESTATE.OUTPUTS;
            } else if (line.startsWith("DEFUZZIFY")) {
                String[] cs = line.split("\\s+");
                if (cs.length == 2) {
                    currentOutput = result.getFuzzyOutputVariable(cs[1]);
                }
                state = PARSESTATE.OUTPUTMEMBER;
            } else if (line.startsWith("RULEBLOCK")) {
                String[] cs = line.split("\\s+");
                if ( cs.length == 2){
                    this.currentRuleGroup = Integer.parseInt(cs[1]);
                }
                state = PARSESTATE.RULES;
            }else if ( line.startsWith("END_RULEBLOCK")){
                state = PARSESTATE.IDLE;
            } else if (line.startsWith("TERMS")) {
                String[] cs = line.split(":=");
                String varname = "";

                //System.out.println("cs[0] : " + cs[0]);
                //System.out.println("cs[1]:" + cs[1]);
                FuzzyVariable current = null;
                if (state == PARSESTATE.INPUTMEMBER) {
                    varname = currentInput.getName();
                    current = currentInput;
                } else {
                    varname = currentOutput.getName();
                    current = currentOutput;
                }
                String memberFunctions = cs[1].trim().toLowerCase();
                String[] defs = memberFunctions.split(":");
                String[] msNames = defs[0].substring(1, defs[0].length() - 1).split(",");
                String[] msValues = defs[1].substring(1, defs[1].length() - 1).split("\\s+");
                float[] msIValues = new float[msValues.length];
                for (int i = 0; i < msValues.length; ++i) {
                    msIValues[i] = Float.parseFloat(msValues[i]);
                }
                int vi = 0;
                for (int i = 0; i < msNames.length; ++i) {
                    MemberShip ms;
                    if (i == 0) {
                        ms = new LeftSigmoidMemberShip(msIValues[vi], msIValues[vi + 1], msNames[i]);
                        ++vi;
                    } else if (i == msNames.length - 1) {
                        ms = new RightSigmoidMemberShip(msIValues[vi - 1], msIValues[vi], msNames[i]);
                        ++vi;
                    } else {
                        if (msNames[i].startsWith("#")) {
                            ms = new TrapezoidMemberShip(msIValues[vi - 1], msIValues[vi], msIValues[vi + 1], msIValues[vi + 2], msNames[i].substring(1));
                            vi += 2;
                        } else {
                            ms = new SigmoidMemberShip(msIValues[vi - 1], msIValues[vi], msIValues[vi + 1], msNames[i]);
                            ++vi;
                        }
                    }
                    current.addMemberShip(ms);
                }
            } else if (line.startsWith("TERM")) {
                String[] cs = line.split(":=");

                //System.out.println("cs[0] : " + cs[0]);
                //System.out.println("cs[1]:" + cs[1]);
                FuzzyVariable current = null;
                if (state == PARSESTATE.INPUTMEMBER) {
                    //varname = currentInput.getName();
                    current = currentInput;
                } else {
                    //varname = currentOutput.getName();
                    current = currentOutput;
                }

                MemberShip ms = null;
                if (cs.length == 2) {
                    String name = cs[0].substring(5).trim();

                    String memberFunction = cs[1].trim().toLowerCase();
                    if (memberFunction.startsWith("triangular")) {
                        int firstIndex = memberFunction.indexOf('(');
                        int secondIndex = memberFunction.indexOf(')');

                        String[] lcr = memberFunction.substring(firstIndex + 1, secondIndex).split("\\s+");
                        float left = Float.parseFloat(lcr[0]);
                        float center = Float.parseFloat(lcr[1]);
                        float right = Float.parseFloat(lcr[2]);

                        ms = new SigmoidMemberShip(left, center, right, name);
                    } else if (memberFunction.startsWith("shoulder left")) {
                        int firstIndex = memberFunction.indexOf('(');
                        int secondIndex = memberFunction.indexOf(')');

                        String[] lcr = memberFunction.substring(firstIndex + 1, secondIndex).split("\\s+");
                        float center = Float.parseFloat(lcr[0]);
                        float right = Float.parseFloat(lcr[1]);
                        ms = new LeftSigmoidMemberShip(center, right, name);

                    } else if (memberFunction.startsWith("shoulder right")) {
                        int firstIndex = memberFunction.indexOf('(');
                        int secondIndex = memberFunction.indexOf(')');

                        String[] lcr = memberFunction.substring(firstIndex + 1, secondIndex).split("\\s+");
                        float left = Float.parseFloat(lcr[0]);
                        float center = Float.parseFloat(lcr[1]);

                        ms = new RightSigmoidMemberShip(left, center, name);
                    } else {
                        // maybe it is a singleon
                        int indexOfSemi = memberFunction.indexOf(';');
                        if (indexOfSemi > -1) {
                            memberFunction = memberFunction.substring(0, indexOfSemi);
                        }
                        float singleton = Float.parseFloat(memberFunction);
                        ms = new SingletonMemberShip(name, singleton);
                    }
                }
                current.addMemberShip(ms);

            } else if (line.startsWith("RULE")) {
                String cs[] = line.split(":");
                if (cs.length >= 2) {
                    result.addFuzzyRule(cs[1], currentRuleGroup);
                }
            } else {
                switch (state) {
                    case INPUTS: {
                        String[] cs = line.split(":");
                        if (cs.length > 0) {
                            FuzzyVariable var = new FuzzyVariable(cs[0]);
                            var.setAsInput();
                            result.addFuzzyInput(var);
                        }
                    }
                    break;
                    case OUTPUTS: {
                        String[] cs = line.split(":");
                        if (cs.length > 0) {
                            FuzzyVariable var = new FuzzyVariable(cs[0]);
                            var.setAsOutput();
                            result.addFuzzyOutput(var);
                        }
                    }
                    break;
                }
            }
        }
    }

    public FuzzySystem getResult() {
        return result;
    }

    public static void main(String[] args) {

        ParseFCL parseFCL = new ParseFCL("./behaviours/takebehaviour/elbowturnbackup.fcl");
        FuzzySystem result = parseFCL.getResult();

//        WriteFCL writeFCL = new WriteFCL("./behaviours/takebehaviour/elbowturnbackup.fcl");
//        writeFCL.write(result);

        FuzzyFrame frame = new FuzzyFrame();
        frame.setFuzzySystem(result);

        frame.setVisible(true);

    }
}
