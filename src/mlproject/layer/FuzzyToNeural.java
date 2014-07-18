/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer;

import intersect.data.fmatrix;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import mlproject.fuzzy.FuzzyRule;
import mlproject.fuzzy.FuzzySystem;
import mlproject.fuzzy.FuzzyVariable;
import mlproject.fuzzy.LeftSigmoidMemberShip;
import mlproject.fuzzy.MemberShip;
import mlproject.fuzzy.RightSigmoidMemberShip;
import mlproject.fuzzy.SigmoidMemberShip;
import mlproject.fuzzy.SingletonMemberShip;
import mlproject.layer.gui.MultiLayerFrame;

/**
 * Initializes a neural network with the data of a fuzzy logic system.
 * @author Koen
 */
public class FuzzyToNeural {

    public static MultiLayerNN FuzzyToNeural(FuzzySystem fuzzySystem) {

        MultiLayerNN neuralFuzzyController = new MultiLayerNN();

        // add a bias term for each cell.
        NeuralNetworkLayer inputLayer = new NeuralNetworkLayer(fuzzySystem.getNrOfInputs() + 1, true, ActivationFunction.LINEAR);
        int i = 0;
        for (FuzzyVariable input : fuzzySystem.getInputs()) {
            inputLayer.setCellName(i, input.getName());
            inputLayer.getCellAt(i).setMaxWeight(-0.5f);
            inputLayer.getCellAt(i).setMinWeight(-20.0f);
            inputLayer.getCellAt(i).setMinimumInput(input.getMinimum());
            inputLayer.getCellAt(i).setMaximumInput(input.getMaximum());
            ++i;
        }
        NeuralNetCell bias = inputLayer.getLastCell();
        bias.setMaxWeight(100);
        bias.setMinWeight(-100);
        neuralFuzzyController.addLayer(inputLayer);

        // first layer shifts and scales the variables to create the correct inputs
        // for the standard sigmoid functions, in other words, the information must
        // be stored into the weights for backpropagation to work.

        // also, the membership functions are coupled (sum of membership values
        // must be one at each point, so the output of these cells must be correctly
        // sent to the sigmoid functions.

        NeuralNetworkLayer memberShipLayer = new NeuralNetworkLayer(true, ActivationFunction.SIGMOID);
        memberShipLayer.setMutable(true);
        memberShipLayer.setInputLayer(inputLayer);

        neuralFuzzyController.addLayer(memberShipLayer);

        ArrayList<InputRange> inputRanges = new ArrayList<>();
        HashMap<MemberShip, MemberShipCellMap> cellMapper = new HashMap<>();

        int inputIndex = 0;
        int biasIndex = inputLayer.getNrOfCells() - 1;
        for (FuzzyVariable input : fuzzySystem.getInputs()) {
            inputRanges.clear();
            SigmoidMemberShipValidator smsv = new SigmoidMemberShipValidator(inputIndex, biasIndex);
            memberShipLayer.addWeightValidator(input.getName(), smsv);
            for (MemberShip ms : input.getMemberShips()) {
                if (ms instanceof LeftSigmoidMemberShip) {
                    LeftSigmoidMemberShip lsm = (LeftSigmoidMemberShip) ms;
                    float x1 = lsm.getCenter();
                    float x2 = lsm.getRight();

                    NeuralNetCell cell = checkInputRange(x1, x2, lsm.getA(), lsm.getB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer);
                    cell.setName(ms.getName());
                    memberShipLayer.addCellToGroup(input.getName(), cell);
                    MemberShipCellMap cellmap = new MemberShipCellMap(memberShipLayer.getIndexOf(cell), -1);
                    cellMapper.put(ms, cellmap);

                } else if (ms instanceof RightSigmoidMemberShip) {
                    RightSigmoidMemberShip rsm = (RightSigmoidMemberShip) ms;
                    float x1 = rsm.getLeft();
                    float x2 = rsm.getCenter();

                    NeuralNetCell cell = checkInputRange(x1, x2, rsm.getA(), rsm.getB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer);
                    memberShipLayer.addCellToGroup(input.getName(), cell);
                    cell.setName(ms.getName());

                    MemberShipCellMap cellmap = new MemberShipCellMap(memberShipLayer.getIndexOf(cell), -1);
                    cellMapper.put(ms, cellmap);
                } else if (ms instanceof SigmoidMemberShip) {
                    SigmoidMemberShip sm = (SigmoidMemberShip) ms;

                    float x1 = sm.getLeft();
                    float x2 = sm.getCenter();
                    float x3 = sm.getRight();

                    NeuralNetCell cell1 = checkInputRange(x1, x2, sm.getLeftA(), sm.getLeftB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer);
                    memberShipLayer.addCellToGroup(input.getName(), cell1);
                    cell1.setName(ms.getName() + "_left");

                    NeuralNetCell cell2 = checkInputRange(x2, x3, sm.getRightA(), sm.getRightB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer);
                    memberShipLayer.addCellToGroup(input.getName(), cell2);
                    cell2.setName(ms.getName() + "_right");

                    MemberShipCellMap cellmap = new MemberShipCellMap(memberShipLayer.getIndexOf(cell1), memberShipLayer.getIndexOf(cell2));
                    cellMapper.put(ms, cellmap);
                }
            }
            ++inputIndex;
        }

        // combine the outputs of the second layer to produce the correct membership values.

        NeuralNetworkLayer memberShipLayer2 = new NeuralNetworkLayer(true, ActivationFunction.LINEAR);
        memberShipLayer2.setInputLayer(memberShipLayer);
        biasIndex = memberShipLayer.getNrOfCells() - 1;
        for (FuzzyVariable input : fuzzySystem.getInputs()) {
            for (MemberShip ms : input.getMemberShips()) {
                NeuralNetCell cell = memberShipLayer2.addCell();
                MemberShipCellMap mscm = cellMapper.get(ms);

                cell.setName(input.getName() + "_" + ms.getName());

                if (ms instanceof LeftSigmoidMemberShip) {
                    cell.setWeight(biasIndex, 1);
                    cell.setWeight(mscm.index1, -1);
                } else if (ms instanceof SigmoidMemberShip) {
                    SigmoidMemberShip sms = (SigmoidMemberShip) ms;
                    cell.setWeight(mscm.index1, 1);
                    cell.setWeight(mscm.index2, -1);
                } else if (ms instanceof RightSigmoidMemberShip) {
                    cell.setWeight(mscm.index1, 1);
                }
            }
        }

        neuralFuzzyController.addLayer(memberShipLayer2);

        // Create a layer to calculate the output of the rules
        NeuralNetworkLayer ruleLayer1 = new NeuralNetworkLayer(false, ActivationFunction.SIGMOID);
        ruleLayer1.setInputLayer(memberShipLayer2);


        for (FuzzyRule rule : fuzzySystem.getRules()) {
            // the fuzzy rule always has one output
            SingleLayerNN ruleAsNeuralNet = trainFuzzyRule(rule);
            String name = rule.getOutputVariable() + "_" + rule.getOutputMemberShip();
            NeuralNetCell ruleCell = ruleLayer1.getCellWithName(name);
            if (ruleCell == null) {
                ruleCell = ruleLayer1.addCell();
                ruleCell.setName(name);
            }

            NeuralNetCell outputCell = ruleAsNeuralNet.getOutputCellAt(0);

            // last cell of the single layer network is a bias term
            for (int ii = 0; ii < ruleAsNeuralNet.getNrOfInputCells() - 1; ++ii) {
                NeuralNetCell nnc = ruleAsNeuralNet.getInputCellAt(ii);
                int cellIndex = memberShipLayer2.getCellIndexWithName(nnc.getName());
                ruleCell.setWeight(cellIndex, outputCell.getWeight(ii));

                System.out.println("");
            }

            ruleCell.setWeight(memberShipLayer2.getNrOfCells() - 1, outputCell.getWeight(ruleAsNeuralNet.getNrOfInputCells() - 1));
        }

        neuralFuzzyController.addLayer(ruleLayer1);

        NeuralNetworkLayer crispLayer = new NeuralNetworkLayer(false, ActivationFunction.LINEAR);
        crispLayer.setMutable(true);

        crispLayer.setInputLayer(ruleLayer1);
        crispLayer.setName("Crisp");
        for (FuzzyVariable output : fuzzySystem.getOutputs()) {
            NeuralNetCell nnc = crispLayer.addCell();
            for (MemberShip ms : output.getMemberShips()) {
                if (ms instanceof SingletonMemberShip) {
                    SingletonMemberShip sms = (SingletonMemberShip) ms;
                    String name = output.getName() + "_" + ms.getName();
                    int index = ruleLayer1.getCellIndexWithName(name);
                    nnc.setWeight(index, sms.getValue());
                }
            }
        }
        neuralFuzzyController.addLayer(crispLayer);

        return neuralFuzzyController;



    }

    public static SingleLayerNN trainFuzzyRule(FuzzyRule rule) {
        ArrayList<MemberShip> collectedInputs = rule.getInputs();
        // add bias term
        int isize = collectedInputs.size() + 1;
        SingleLayerNN layer = new SingleLayerNN(isize, 1, ActivationFunction.SIGMOID);
        for (int i = 0; i < collectedInputs.size(); ++i) {
            MemberShip ms = collectedInputs.get(i);
            String cellName = ms.getParent().getName() + "_" + ms.getName();
            layer.setInputCellName(i, cellName);
        }

        layer.initialize(-0.2f, 0.2f);
        Random r = new Random(System.currentTimeMillis());
        fmatrix inputs = new fmatrix(isize, 1);
        inputs.set(isize, 1, 1);

        fmatrix expectedOutputs = new fmatrix(1, 1);
        for (int i = 0; i < 1000000; ++i) {
            if (r.nextBoolean()) {
                for (int j = 0; j < collectedInputs.size(); ++j) {
                    float value = r.nextFloat();
                    inputs.set(j + 1, 1, value);
                    collectedInputs.get(j).setInputTestValue(value);
                }
            } else {
                for (int j = 0; j < collectedInputs.size(); ++j) {
                    float value = r.nextInt(2);
                    inputs.set(j + 1, 1, value);
                    collectedInputs.get(j).setInputTestValue(value);
                }
            }
            float result = rule.evaluateTestValue();
            expectedOutputs.set(1, 1, result);
            layer.forward(inputs);
            layer.backpropagate(-0.005f, expectedOutputs);
        }
        layer.printOutputLayer();

        /*
        int nrOfErrors = 0;
        for (int i = 0; i < 1000; ++i){
        for (int j = 0; j < collectedInputs.size();++j)
        {
        float value = r.nextFloat();
        inputs.set(j+1,1,value);
        collectedInputs.get(j).setInputTestValue(value);
        }
        
        float result = rule.evaluateTestValue();
        
        layer.forward(inputs);
        float result2 = layer.getOutput(0);
        }*/

        return layer;
    }

    /**
     * Alternative for the membership functions
     * @param fuzzySystem the fuzzySystem to convert.
     * @return the new neural network for the fuzzy system
     */
    public static MultiLayerNN FuzzyToNeural2(FuzzySystem fuzzySystem) {
        if ( fuzzySystem == null)
            return null;
        MultiLayerNN neuralFuzzyController = new MultiLayerNN();

        // add a bias term for each cell.
        NeuralNetworkLayer inputLayer = new NeuralNetworkLayer(fuzzySystem.getNrOfInputs() + 1, true, ActivationFunction.LINEAR);
        int i = 0;
        for (FuzzyVariable input : fuzzySystem.getInputs()) {
            inputLayer.setCellName(i, input.getName());
            
            inputLayer.getCellAt(i).setMinimumInput(input.getMinimum());
            inputLayer.getCellAt(i).setMaximumInput(input.getMaximum());
            ++i;
        }
        NeuralNetCell bias = inputLayer.getLastCell();
       
        neuralFuzzyController.addLayer(inputLayer);

        // first layer shifts and scales the variables to create the correct inputs
        // for the standard sigmoid functions, in other words, the information must
        // be stored into the weights for backpropagation to work.

        // also, the membership functions are coupled (sum of membership values
        // must be one at each point, so the output of these cells must be correctly
        // sent to the sigmoid functions.

        NeuralNetworkLayer memberShipLayer_b = new NeuralNetworkLayer(false, ActivationFunction.LINEAR);
        memberShipLayer_b.setMutable(true);
        memberShipLayer_b.setInputLayer(inputLayer);
        neuralFuzzyController.addLayer(memberShipLayer_b);



        ArrayList<InputRange> inputRanges = new ArrayList<>();
        HashMap<MemberShip, MemberShipCellMap> cellMapper = new HashMap<>();

        int inputIndex = 0;
        int biasIndex = inputLayer.getNrOfCells() - 1;
        for (FuzzyVariable input : fuzzySystem.getInputs()) {
            inputRanges.clear();
            SigmoidMemberShipValidator smsv = new SigmoidMemberShipValidator(inputIndex, biasIndex);
            memberShipLayer_b.addWeightValidator(input.getName(), smsv);
            for (MemberShip ms : input.getMemberShips()) {
                if (ms instanceof LeftSigmoidMemberShip) {
                    LeftSigmoidMemberShip lsm = (LeftSigmoidMemberShip) ms;
                    float x1 = lsm.getCenter();
                    float x2 = lsm.getRight();

                    NeuralNetCell cell_b = checkInputRange2(x1, x2, lsm.getA(), lsm.getB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer_b);
                    cell_b.addAlias(input.getName()+ "_" + ms.getName() + "_b");
                    memberShipLayer_b.addCellToGroup(input.getName(), cell_b);
                } else if (ms instanceof RightSigmoidMemberShip) {
                    RightSigmoidMemberShip rsm = (RightSigmoidMemberShip) ms;
                    float x1 = rsm.getLeft();
                    float x2 = rsm.getCenter();

                    NeuralNetCell cell = checkInputRange2(x1, x2, rsm.getA(), rsm.getB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer_b);
                    memberShipLayer_b.addCellToGroup(input.getName(), cell);
                    cell.addAlias(input.getName()+"_" + ms.getName()+"_b");
                } else if (ms instanceof SigmoidMemberShip) {
                    SigmoidMemberShip sm = (SigmoidMemberShip) ms;

                    float x1 = sm.getLeft();
                    float x2 = sm.getCenter();
                    float x3 = sm.getRight();

                    NeuralNetCell cell1 = checkInputRange2(x1, x2, sm.getLeftA(), sm.getLeftB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer_b);
                    memberShipLayer_b.addCellToGroup(input.getName(), cell1);
                    cell1.addAlias(input.getName() + "_" + ms.getName() + "_left_b");

                    NeuralNetCell cell2 = checkInputRange2(x2, x3, sm.getRightA(), sm.getRightB(), inputIndex, biasIndex,
                            inputRanges, inputLayer, memberShipLayer_b);
                    memberShipLayer_b.addCellToGroup(input.getName(), cell2);
                    cell2.addAlias(input.getName() + "_" + ms.getName() + "_right_b");
                }
            }
            ++inputIndex;
        }
        // now that the b weights, have been applied, a sigmoid layer follows that
        // applies the a-weights :
        NeuralNetworkLayer memberShipLayer_a = new NeuralNetworkLayer(memberShipLayer_b.getNrOfCells()+1,true, ActivationFunction.SIGMOID);
        memberShipLayer_a.setMutable(true);
        memberShipLayer_a.setInputLayer(memberShipLayer_b);
        neuralFuzzyController.addLayer(memberShipLayer_a);

        for (FuzzyVariable input : fuzzySystem.getInputs()) {
            inputRanges.clear();
            for (MemberShip ms : input.getMemberShips()) {
                if (ms instanceof LeftSigmoidMemberShip) {
                    LeftSigmoidMemberShip lsm = (LeftSigmoidMemberShip)ms;
                    String name = input.getName()+ "_" + ms.getName() + "_b";
                    NeuralNetCell cell  = memberShipLayer_b.getCellByAlias(name);
                    int index = memberShipLayer_b.getIndexOf(cell);
                    
                    NeuralNetCell acell =  memberShipLayer_a.getCellAt(index);
                    acell.setWeight(index, lsm.getA());
                    acell.setAllInputWeightsMutable(false);
                    acell.setInputWeightMutable(index,true);
                    
                    MemberShipCellMap cellmap = new MemberShipCellMap(index, -1);
                    cellMapper.put(ms, cellmap);

                } else if (ms instanceof RightSigmoidMemberShip) {
                    RightSigmoidMemberShip rsm = (RightSigmoidMemberShip) ms;
                    String name = input.getName()+ "_" + ms.getName() + "_b";
                    NeuralNetCell cell  = memberShipLayer_b.getCellByAlias(name);
                    int index = memberShipLayer_b.getIndexOf(cell);
                    
                    NeuralNetCell acell = memberShipLayer_a.getCellAt(index);
                    acell.setWeight(index, rsm.getA());
                    acell.setAllInputWeightsMutable(false);
                    acell.setInputWeightMutable(index,true);
                   
                    MemberShipCellMap cellmap = new MemberShipCellMap(index, -1);
                    cellMapper.put(ms, cellmap);
                } else if (ms instanceof SigmoidMemberShip) {
                    SigmoidMemberShip sm = (SigmoidMemberShip) ms;
                    String leftName = input.getName() + "_" + ms.getName() + "_left_b";
                    String rightName = input.getName() + "_" +ms.getName() + "_right_b";
                    
                    NeuralNetCell left_cell = memberShipLayer_b.getCellByAlias(leftName);
                    int index_left = memberShipLayer_b.getIndexOf(left_cell);
                    NeuralNetCell right_cell = memberShipLayer_b.getCellByAlias(rightName);
                    int index_right = memberShipLayer_b.getIndexOf(right_cell);
                    
                    NeuralNetCell a_leftCell  = memberShipLayer_a.getCellAt(index_left);
                    a_leftCell.setWeight(index_left, sm.getLeftA());
                    a_leftCell.setAllInputWeightsMutable(false);
                    a_leftCell.setInputWeightMutable(index_left,true);
                    
                    NeuralNetCell a_rightCell = memberShipLayer_a.getCellAt(index_right);
                    a_rightCell.setWeight(index_right, sm.getRightA());
                    a_rightCell.setAllInputWeightsMutable(false);
                    a_rightCell.setInputWeightMutable(index_right,true);
                    a_rightCell.setMaxWeight(-0.5f);
                    a_rightCell.setMinWeight(-20.0f);
                    
                    MemberShipCellMap cellmap = new MemberShipCellMap(index_left,index_right);
                    cellMapper.put(ms, cellmap);
                }
            }
            
            
            ++inputIndex;
        }

        // combine the outputs of the second layer to produce the correct membership values.

        NeuralNetworkLayer memberShipLayer2 = new NeuralNetworkLayer(true, ActivationFunction.LINEAR);
        memberShipLayer2.setInputLayer(memberShipLayer_a);
        biasIndex = memberShipLayer_a.getNrOfCells() - 1;
        for (FuzzyVariable input : fuzzySystem.getInputs()) {
            for (MemberShip ms : input.getMemberShips()) {
                NeuralNetCell cell = memberShipLayer2.addCell();
                MemberShipCellMap mscm = cellMapper.get(ms);

                cell.setName(input.getName() + "_" + ms.getName());
                if (ms instanceof LeftSigmoidMemberShip) {
                    cell.setWeight(biasIndex, 1);
                    cell.setWeight(mscm.index1, -1);
                } else if (ms instanceof SigmoidMemberShip) {
                    cell.setWeight(mscm.index1, 1);
                    cell.setWeight(mscm.index2, -1);
                } else if (ms instanceof RightSigmoidMemberShip) {
                    cell.setWeight(mscm.index1, 1);
                }
            }
        }

        neuralFuzzyController.addLayer(memberShipLayer2);

        // Create a layer to calculate the output of the rules
        NeuralNetworkLayer ruleLayer1 = new NeuralNetworkLayer(false, ActivationFunction.SIGMOID);
        ruleLayer1.setInputLayer(memberShipLayer2);


        for (FuzzyRule rule : fuzzySystem.getRules()) {
            // the fuzzy rule always has one output
            SingleLayerNN ruleAsNeuralNet = trainFuzzyRule(rule);
            String name = rule.getOutputVariable() + "_" + rule.getOutputMemberShip();
            NeuralNetCell ruleCell = ruleLayer1.getCellWithName(name);
            if (ruleCell == null) {
                ruleCell = ruleLayer1.addCell();
                ruleCell.setName(name);
            }

            NeuralNetCell outputCell = ruleAsNeuralNet.getOutputCellAt(0);

            // last cell of the single layer network is a bias term
            for (int ii = 0; ii < ruleAsNeuralNet.getNrOfInputCells() - 1; ++ii) {
                NeuralNetCell nnc = ruleAsNeuralNet.getInputCellAt(ii);
                int cellIndex = memberShipLayer2.getCellIndexWithName(nnc.getName());
                ruleCell.setWeight(cellIndex, outputCell.getWeight(ii));

                System.out.println("");
            }

            ruleCell.setWeight(memberShipLayer2.getNrOfCells() - 1, outputCell.getWeight(ruleAsNeuralNet.getNrOfInputCells() - 1));
        }

        neuralFuzzyController.addLayer(ruleLayer1);

        NeuralNetworkLayer crispLayer = new NeuralNetworkLayer(false, ActivationFunction.LINEAR);
        crispLayer.setMutable(true);

        crispLayer.setInputLayer(ruleLayer1);
        crispLayer.setName("Crisp");
        for (FuzzyVariable output : fuzzySystem.getOutputs()) {
            NeuralNetCell nnc = crispLayer.addCell();
            for (MemberShip ms : output.getMemberShips()) {
                if (ms instanceof SingletonMemberShip) {
                    SingletonMemberShip sms = (SingletonMemberShip) ms;
                    String name = output.getName() + "_" + ms.getName();
                    int index = ruleLayer1.getCellIndexWithName(name);
                    nnc.setWeight(index, sms.getValue());
                }
            }
        }
        neuralFuzzyController.addLayer(crispLayer);
        
        return neuralFuzzyController;
    }

    public static void evaluate(MultiLayerNN neuralFuzzyController, String name, float input) {
        fmatrix inputs = new fmatrix(1, 1);
        inputs.set(1, 1, input);

        neuralFuzzyController.forward(inputs);
        System.out.println("inputs : ");
        neuralFuzzyController.printLayer(0);
        System.out.println("layer 1");
        neuralFuzzyController.printLayer(1);
        System.out.println("layer 2");
        neuralFuzzyController.printLayer(2);
        System.out.println("layer 3");
        neuralFuzzyController.printLayer(3);
        System.out.println("layer 4");
        neuralFuzzyController.printLayer(4);

    }

    private static NeuralNetCell checkInputRange(float x1, float x2, float A, float B,
            int inputIndex, int biasIndex,
            ArrayList<InputRange> inputRanges,
            NeuralNetworkLayer inputLayer,
            NeuralNetworkLayer memberShipLayer) {
        InputRange ir = new InputRange(x1, x2);
        NeuralNetCell cell;
        if (!inputRanges.contains(ir)) {
            cell = memberShipLayer.addCell(inputLayer.getNrOfCells());
            ir.setCell(cell);
            inputRanges.add(ir);
            cell.setWeight(inputIndex, A);
            cell.setWeight(biasIndex, -B * A);
        } else {
            int index = inputRanges.indexOf(ir);
            cell = inputRanges.get(index).getCell();
        }
        return cell;
    }

    private static NeuralNetCell checkInputRange2(float x1, float x2, float A, float B,
            int inputIndex, int biasIndex,
            ArrayList<InputRange> inputRanges,
            NeuralNetworkLayer inputLayer,
            NeuralNetworkLayer memberShipLayer_b) {
        InputRange ir = new InputRange(x1, x2);
        NeuralNetCell cell_b;
        if (!inputRanges.contains(ir)) {
            cell_b = memberShipLayer_b.addCell(inputLayer.getNrOfCells());
            ir.setCell(cell_b);
            inputRanges.add(ir);
            cell_b.setWeight(inputIndex, 1);
            cell_b.setInputWeightMutable(inputIndex,false);
            cell_b.setWeight(biasIndex, -B);
            
            cell_b.setMinWeight(-20.0f);
            cell_b.setMaxWeight(-0.1f);
        } else {
            int index = inputRanges.indexOf(ir);
            cell_b = inputRanges.get(index).getCell();
        }
        return cell_b;
    }

    private static int checkInputRange3(float x1, float x2,
            ArrayList<InputRange> inputRanges,
            NeuralNetworkLayer memberShipLayer_b) {
        InputRange ir = new InputRange(x1, x2);
        int index = inputRanges.indexOf(ir);
        if (index != -1) {
            NeuralNetCell cell = inputRanges.get(index).getCell();
            return memberShipLayer_b.getIndexOf(cell);
        }else
            return -1;
    }

    public static void main(String[] args) {
        FuzzySystem system = new FuzzySystem("neural test");
        FuzzyVariable iv = new FuzzyVariable("input");
        iv.addMemberShip(new LeftSigmoidMemberShip(-10, -5, "left"));
        iv.addMemberShip(new SigmoidMemberShip(-10, -5, 1, "center"));
        iv.addMemberShip(new RightSigmoidMemberShip(-5, 1, "right"));
        system.addFuzzyInput(iv);

        FuzzyVariable ov = new FuzzyVariable("output");
        ov.addMemberShip(new SingletonMemberShip("fast", 1));
        ov.addMemberShip(new SingletonMemberShip("slow", 2));

        system.addFuzzyOutput(ov);

        system.addFuzzyRule("if input is left && input is center then output is fast");
        system.addFuzzyRule("if input is right then output is slow");

        MultiLayerNN ml = FuzzyToNeural.FuzzyToNeural(system);



        evaluate(ml, "test", -6.1f);
        iv.setCurrentValue(-6.1f);
        system.evaluate();
        System.out.println("Result of fuzzy system : " + ov.getOutputValue());


        float value1 = iv.evaluateAntecedent("left", -6.1f, false);
        float value2 = iv.evaluateAntecedent("center", -6.1f, false);
        float value3 = iv.evaluateAntecedent("right", -6.1f, false);
        System.out.println("Fuzzy outputs : ");
        System.out.println("value 0:" + value1);
        System.out.println("value 1:" + value2);
        System.out.println("value 2:" + value3);


        MultiLayerFrame frame = new MultiLayerFrame();
        frame.setMultiLayer(ml);
        frame.engage();
    }
}

class InputRange {

    private float x1;
    private float x2;
    // the cell that outputs the correct value for the sigmoid function.
    private NeuralNetCell cell;

    public InputRange(float x1, float x2) {
        this.x1 = x1;
        this.x2 = x2;
    }

    public void setCell(NeuralNetCell cell) {
        this.cell = cell;
    }

    @Override
    public boolean equals(Object second) {
        if (second instanceof InputRange) {
            InputRange ir = (InputRange) second;
            float delta1 = Math.abs(ir.x1 - this.x1);
            float delta2 = Math.abs(ir.x2 - this.x2);
            return (delta1 < 1e-5 && delta2 < 1e-5);
        } else {
            return false;
        }
    }

    public NeuralNetCell getCell() {
        return cell;
    }
}

class MemberShipCellMap {

    public MemberShipCellMap(int index1, int index2) {
        this.index1 = index1;
        this.index2 = index2;
    }
    public int index1;
    public int index2;
}