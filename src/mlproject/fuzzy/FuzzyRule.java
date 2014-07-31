/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.fuzzy;

import java.util.ArrayList;
import java.util.Stack;
import mlproject.fuzzy.logic.And;
import mlproject.fuzzy.logic.Antecedent;
import mlproject.fuzzy.logic.Group;
import mlproject.fuzzy.logic.Or;
import mlproject.fuzzy.logic.RulePart;

/**
 *
 * @author Koen
 */
public class FuzzyRule implements Comparable {

    private FuzzySystem parentSystem;
    private String rule;
    private String outputVariable;
    private String outputMemberShip;
    // for bookkeeping purposes
    private ArrayList<Antecedent> antecedentList = new ArrayList<Antecedent>();
    private boolean isParsed = false;
    private String parseError;

    public ArrayList<MemberShip> getInputs() {
        ArrayList<MemberShip> inputs = new ArrayList<>();
        top.getInputs(parentSystem, inputs);
        return inputs;
    }

    /**
     * @return the parentSystem
     */
    public FuzzySystem getParentSystem() {
        return parentSystem;
    }

    /**
     * @param parentSystem the parentSystem to set
     */
    public void setParentSystem(FuzzySystem parentSystem) {
        this.parentSystem = parentSystem;
    }

    public void setOutputVar(String outputVar) {
        this.outputVariable = outputVar;
    }

    @Override
    public int compareTo(Object o) {
        FuzzyRule other = (FuzzyRule) o;
        return (int) ((other.getCurrentResult() - this.getCurrentResult()) * 10);
    }

    public boolean isParsed() {
        return isParsed;
    }
    
    public String getParseError(){
        return parseError;
    }

   

    private enum ParseState {

        IDLE, ANTECEDENT, OR, AND, NOT, STARTGROUP, ENDGROUP
    };
    private ParseState currentState = ParseState.IDLE;
    private Stack<RulePart> parseStack = new Stack();
    private float currentResult;
    // the top level rule part
    private RulePart top;

    /**
     * Initializes the rule with a string
     *
     * @param rule
     */
    public FuzzyRule(String rule) {
        parseRule(rule);
    }

    public String getRuleText() {
        return rule;
    }

    public RulePart getTopRulePart() {
        return top;
    }

    private void parseRule(String rule) {
        this.rule = rule.toLowerCase();

        int indexOfThen = rule.indexOf("then");
        int indexOfIf = rule.indexOf("if");

        if (indexOfThen < 0 || indexOfIf < 0) {
            isParsed = false;
            if ( indexOfIf < 0){
                parseError = "No antecedent found";
                return;
            }else if ( indexOfThen < 0 ){
                parseError = "No conclusion found";
                return;
            }
        }

        boolean success1  = parseAnteCedent(rule.substring(indexOfIf + 2, indexOfThen));

        boolean success2 = parseConclusion(rule.substring(indexOfThen + 4));
        
        isParsed = success1 && success2;
    }

    private boolean parseAnteCedent(String antecedent) {
        System.out.println(antecedent);

        // count number of left and right braces.
        // count left braces
        int index = -1;
        int leftCount = 0;
        do {
            index = antecedent.indexOf('(', index + 1);
        } while (index > 0);

        index = -1;
        int rightCount = 0;

        do {
            index = antecedent.indexOf(')', index + 1);
        } while (index > 0);

        if (leftCount != rightCount) {
            System.out.println("braces do not match !");
            parseError = "Unmatched braces.";
            return false;
        }

        // start parsing.
        for (int i = 0; i < antecedent.length(); ++i) {
            char current = antecedent.charAt(i);

            if (Character.isWhitespace(current)) {
                continue;
            } else if (current == '(') {
                currentState = ParseState.STARTGROUP;
                Group g = new Group();
                pushGroup(g);

            } else if (current == ')') {
                currentState = ParseState.ENDGROUP;
                popGroup();
            } else if (current == '&') {
                if (antecedent.charAt(i + 1) == '&') {
                    ++i;
                    And a = new And();
                    pushAnd(a);
                }

            } else if (current == '|') {
                if (antecedent.charAt(i + 1) == '|') {
                    ++i;
                    Or o = new Or();
                    pushOr(o);
                }
            } else if (Character.isLetter(current) || current == '@' || current == '$') {
                // start antecedent parsing
                // need to parse three words.
                Antecedent currentAntecedent = new Antecedent();

                String variable = readString(antecedent, i);
                i += variable.length();
                i = skipWhiteSpace(antecedent, i);
                String is = readString(antecedent, i);
                i += is.length();
                i = skipWhiteSpace(antecedent, i);
                String membership = readString(antecedent, i);
                i += membership.length();
                currentAntecedent.setVariable(variable);
                currentAntecedent.setMemberShip(membership);
                currentAntecedent.setNegation(is.equals("isnot"));
                antecedentList.add(currentAntecedent);
                pushAntecedent(currentAntecedent);
            }
        }
        top = parseStack.firstElement();
        return true;
    }

    public int skipWhiteSpace(String toSkip, int startindex) {
        while (Character.isWhitespace(toSkip.charAt(startindex))) {
            startindex++;
        }
        return startindex;
    }

    public String readString(String toRead, int startindex) {
        int endindex = startindex;
        while (Character.isLetterOrDigit(toRead.charAt(endindex)) || toRead.charAt(endindex) == '_' 
                || toRead.charAt(endindex) == '@' || toRead.charAt(endindex) == '$') {
            endindex++;
        }
        return toRead.substring(startindex, endindex);
    }

    public static void main(String[] args) {
        String rule = "if (angle is right) && (distance is near) then speed is up";
        FuzzyRule fr = new FuzzyRule(rule);
    }

    public Iterable<Antecedent> getAntecedents() {
        return antecedentList;
    }

    /**
     * An antecedent
     *
     * @param a
     */
    public void pushAntecedent(Antecedent a) {
        if (parseStack.size() > 0) {
            RulePart rp = parseStack.peek();
            if (rp instanceof Group) {
                Group g = (Group) rp;
                g.setChild(a);
            } else if (rp instanceof And) {
                And and = (And) rp;
                and.addOperator(a);
            } else if (rp instanceof Or) {
                Or or = (Or) rp;
                or.addOperator(a);
            }
        } else {
            parseStack.push(a);
        }
    }

    public void pushAnd(And a) {
        if (parseStack.size() > 0) {
            RulePart rp = parseStack.peek();
            if (!(rp instanceof And)) {
                rp = parseStack.pop();
                a.addOperator(rp);
                parseStack.push(a);
            }
        } else {
            parseStack.push(a);
        }
    }

    public void pushOr(Or o) {
        if (parseStack.size() > 0) {
            RulePart rp = parseStack.peek();
            if (!(rp instanceof Or)) {
                rp = parseStack.pop();
                o.addOperator(rp);
                parseStack.push(o);
            }
        }
    }

    public void pushGroup(Group g) {
        if (parseStack.size() > 0) {
            RulePart rp = parseStack.peek();
            if (rp instanceof Or) {
                ((Or) rp).addOperator(g);
            } else if (rp instanceof And) {
                ((And) rp).addOperator(g);
            } else if (rp instanceof Group) {
                ((Group) rp).setChild(g);
            }
        }
        parseStack.push(g);
    }

    public void popGroup() {
    }

    private boolean parseConclusion(String substring) {
        int indexOfIs = substring.indexOf("is ");
        if ( indexOfIs < 0 ){
            return false;
        }
        outputVariable = substring.substring(0, indexOfIs).trim();
        if ( outputVariable.length() ==0 ){
            return false;
        }
        outputMemberShip = substring.substring(indexOfIs + 2).trim();
        if ( outputMemberShip.length() == 0){
            return false;
        }
        
        return true;
    }

    public String getOutputVariable() {
        return outputVariable;
    }

    public String getOutputMemberShip() {
        return outputMemberShip;
    }

    public void evaluate(FuzzySystem system) {

        float result = top.evaluateAntecedent(system);
        FuzzyVariable output = system.getFuzzyOutputVariable(this.outputVariable);
        if (output == null) {
            System.out.println("Output variable not found : " + outputVariable);
            return;
        }
        MemberShip memberShip = output.getMemberShip(this.outputMemberShip);

        if (memberShip instanceof SingletonMemberShip) {
            SingletonMemberShip sms = (SingletonMemberShip) memberShip;
            float value = sms.getValue();
            // add the outputvalue with the weights
            output.addOutputValue(result, value);
        }

        this.currentResult = result;
    }
    
    public void evaluateVerbal(FuzzySystem system) {
        float result = top.evaluateAntecedentVerbally(system);
        
        FuzzyVariable output = system.getFuzzyOutputVariable(this.outputVariable);
        if (output == null) {
            System.out.println("Output variable not found : " + outputVariable);
            return;
        }
        MemberShip memberShip = output.getMemberShip(this.outputMemberShip);

        if (memberShip instanceof SingletonMemberShip) {
            SingletonMemberShip sms = (SingletonMemberShip) memberShip;
            float value = sms.getValue();
            System.out.println("Output : " + sms.getName() + "[weight,value]:" + result +"," + value);
            // add the outputvalue with the weights
            output.addOutputValue(result, value);
        }
        this.currentResult = result;
        
    }

    public float getCurrentResult() {
        return currentResult;
    }

    public float evaluateTestValue() {
        return top.evaluateTestValue(this.parentSystem);
    }
    StringBuilder text = new StringBuilder();

    @Override
    public String toString() {
        text.delete(0, text.length());
        String ruleText = this.rule;
        text.append("<html><body>");
        text.append("<font color='");

        int nrOfVisibleDots = (int) (getCurrentResult() * 10);
        for (int i = 0; i < 10; ++i) {
            if (i <= nrOfVisibleDots) {
                text.append("<font color='#000000'>&#9679;</font>");
            } else {
                text.append("<font color='#ffffff'>&#9679;</font>");
            }
        }
        ruleText = ruleText.replaceAll("if", "<b>if</b>");
        ruleText = ruleText.replaceAll("then", "<b>then</b>");
        ruleText = ruleText.replaceAll("&&", "<b>&&</b>");

        text.append(ruleText);
        text.append("<br/>");
        text.append("</body></html>");
        
        return text.toString();
    }
}
