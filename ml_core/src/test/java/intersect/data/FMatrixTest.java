package intersect.data;

import dae.matrix.Range;
import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import dae.matrix.tmatrix;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class FMatrixTest {

    public FMatrixTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    // TODO add test methods here.
    // The methods must be annotated with annotation @Test. For example:
    //
    @Test
    public void testMatrix() {
        Range r = new Range();
        r.startOfRange = 2.57f;
        r.endOfRange = 11.2f;
        r.increment = 0.2f;
        fmatrix result = fmatrix.construct(r);
        System.out.println(result);

        imatrix tresult = new tmatrix(result);
        System.out.println(tresult);

        fmatrix identity = fmatrix.eye(3, 3);
        System.out.println(identity);

        fmatrix op1 = fmatrix.random(3, 2, -5.0f, 1.0f);
        fmatrix op2 = fmatrix.random(2, 1, 8.0f, 20.0f);

        imatrix result2 = fmatrix.multiply(op1, op2);

        System.out.println(op1);
        System.out.println(op2);
        System.out.println(result2);

        fmatrix test = fmatrix.construct("1:10");
        System.out.println(test);
        System.out.println("Sum is : " + test.sum());
        test.multiply(2.0f);
        System.out.println("Sum is : " + test.sum());

        op1 = fmatrix.random(4, 3, 0, 100);
        op2 = fmatrix.random(2, 5, -50, -10);

        imatrix merge = fmatrix.mergeRows(op1, op2);
        System.out.println(op1);
        System.out.println(op2);
        System.out.println(merge);

        imatrix merge2 = fmatrix.mergeColumns(op1, op2);
        System.out.println(merge2);

        System.out.println("To copy : ");
        System.out.println(op1);
        imatrix op1_copy = op1.copy();
        System.out.println(op1_copy);
        imatrix op1_copy_t = op1.tcopy();
        System.out.println(op1_copy_t);

        System.out.println("addRow test");
        fmatrix rowTest1 = fmatrix.random(2, 2, -1, 2);

        System.out.println("before add");
        System.out.println(rowTest1);
        /*
        //ArrayList<Float> rowToAdd = new ArrayList<>();
        rowToAdd.add(12.22f);
        rowToAdd.add(13.2f);
        rowToAdd.add(17.1f);
        
        rowTest1.addRow(rowToAdd);
        System.out.println("after add");
        System.out.println(rowTest1);
         */

        fmatrix op3 = fmatrix.random(2, 7, -50, -10);
        fmatrix row1 = op3.getRow(1);

        System.out.println("op3:\n" + op3);
        System.out.println("row1 :\n" + row1);
        System.out.println("column2 : \n" + op3.getColumn(2));
    }
}
