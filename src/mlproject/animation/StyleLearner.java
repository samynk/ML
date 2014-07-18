/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.animation;

import intersect.data.fmatrix;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import mlproject.layer.MultiLayerNN;

/**
 * An animation that can be used to train the neural network.
 * @author Koen
 */
public class StyleLearner {
    private HashMap<String,ArrayList<Float>> motionData = new HashMap<>();
    private String[] headers;
    private int nrOfSamples;
    
 
    
    public StyleLearner(File animationWithStyle ){
        try {
            BufferedReader br = new BufferedReader(new FileReader(animationWithStyle));
            String headerLine = br.readLine();
            headers = headerLine.split(";");
            for (String header : headers ){
                motionData.put(header, new ArrayList<Float>());
            }
            String line= null;
            nrOfSamples = 0;
            while ( (line = br.readLine()) != null)
            {
                String[] data = line.split(";");
                for (int i = 0; i < data.length; ++i)
                {
                    ArrayList<Float> dataList = motionData.get(headers[i]);
                    dataList.add(Float.parseFloat(data[i]));
                }
                nrOfSamples++;
            }
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
        
    }

    public StyleLearner(File[] selectedFiles) {
        for (File file : selectedFiles){
            try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String headerLine = br.readLine();
            headers = headerLine.split(";");
            for (String header : headers ){
                if ( !motionData.containsKey(header))
                    motionData.put(header, new ArrayList<Float>());
            }
            String line= null;
            nrOfSamples = 0;
            while ( (line = br.readLine()) != null)
            {
                String[] data = line.split(";");
                for (int i = 0; i < data.length; ++i)
                {
                    ArrayList<Float> dataList = motionData.get(headers[i]);
                    dataList.add(Float.parseFloat(data[i]));
                }
                nrOfSamples++;
            }
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
        }
    }
    
    public float getSample(String name, int index){
        ArrayList<Float> samples = motionData.get(name);
        if ( samples != null)
            return samples.get(index);
        else
            return Float.NaN;
    }
    
    public int getNrOfSamples(){
        return nrOfSamples;
    }
    
    public void adaptNeuralNet(String data, MultiLayerNN toLearn ){
        String rotHeader = data+"_rot";
        String xHeader = data+"_x";
        String yHeader = data+"_y";
        
        ArrayList<Float> rotations = motionData.get(rotHeader);
        ArrayList<Float> limbxList = motionData.get(xHeader);
        ArrayList<Float> limbyList = motionData.get(yHeader);
        
        ArrayList<Float> targetxList = motionData.get("targetx");
        ArrayList<Float> targetyList = motionData.get("targety");
        
        ArrayList<Float> limbtargetxList = motionData.get("limbtargetx");
        ArrayList<Float> limbtargetyList = motionData.get("limbtargety");
        
        Random r = new Random(System.currentTimeMillis());
        fmatrix expectedOutput = new fmatrix(1,1);
        int count = 0;
        for (int i = 0; i < 10000; ++i )
        {
            int index = r.nextInt(this.nrOfSamples-1);
            float rot1 = rotations.get(index);
            float rot2 = rotations.get(index+1);
            
            float limbx = limbxList.get(index);
            float limby = limbyList.get(index);
            
            float targetx = targetxList.get(index);
            float targety = targetyList.get(index);
            
            float limbtargetx = limbtargetxList.get(index);
            float limbtargety = limbtargetyList.get(index);

            double dx1 = targetx - limbx;
            double dy1 = targety - limby;

            double dx2 = limbtargetx - limbx;
            double dy2 = limbtargety - limby;

            float angle = (float) (Math.acos((dx1 * dx2 + dy1 * dy2) / (Math.hypot(dx1, dy1) * Math.hypot(dx2, dy2))) * 180.0 / Math.PI);

            double z = dx2*dy1 - dx1*dy2;
            if ( z > 0 )
                angle = -angle;
           
            float distance = (float) Math.hypot(dx1, dy1);
            float distance2 = (float) Math.hypot(dx2, dy2);

            toLearn.setInput("distance", distance-distance2);
            toLearn.setInput("angle", angle);
            
            
            toLearn.forward();
            float expected = (float)(50*(rot1-rot2)*Math.PI/180.0);
            
            //System.out.println("Distance : " +(distance-distance2));
            //System.out.println("Angle : " + angle);
            //System.out.println("Expected : " + expected);
            //System.out.println("Outputvalue :  "+toLearn.getOutputLayer().getOutput(0));
            expectedOutput.set(1,1,expected);
            
            toLearn.backpropagate(0.001f, expectedOutput);
            count++;
        }
        System.out.println("Learning finished , nr of samples : "+ count);
    }
}