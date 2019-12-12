import es.uam.eps.ir.knnbandit.utils.statistics.GiniIndex;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

/**
 * Tests for the GiniIndex class.
 */
public class GiniTest
{
    @Test
    public void oneStep()
    {
        int numItems = 3;
        GiniIndex gini = new GiniIndex(numItems);

        Assertions.assertTrue(Double.isNaN(gini.getValue()));
        // 1 0 0
        gini.updateFrequency(0, 1);
        Assertions.assertEquals(1.0, gini.getValue(), 1E-4);
        // 2 0 0
        gini.updateFrequency(0, 1);
        Assertions.assertEquals(1.0, gini.getValue(), 1E-4);
        // 3 0 0
        gini.updateFrequency(0, 1);
        Assertions.assertEquals(1.0, gini.getValue(), 1E-4);
        // 3 1 0
        gini.updateFrequency(1, 1);
        Assertions.assertEquals(0.75, gini.getValue(), 1E-4);
        // 3 2 0
        gini.updateFrequency(1, 1);
        Assertions.assertEquals(0.6, gini.getValue(), 1E-4);
        // 3 2 1
        gini.updateFrequency(2, 1);
        Assertions.assertEquals(1.0 / 3.0, gini.getValue(), 1E-4);
        // 2 2 1
        gini.updateFrequency(0, -1);
        Assertions.assertEquals(0.2, gini.getValue(), 1E-4);
        // 2 1 1
        gini.updateFrequency(1, -1);
        Assertions.assertEquals(0.25, gini.getValue(), 1E-4);
        // 1 1 1
        gini.updateFrequency(0, -1);
        Assertions.assertEquals(0.0, gini.getValue(), 1E-4);
        // 1 1 0
        gini.updateFrequency(2, -1);
        Assertions.assertEquals(0.5, gini.getValue(), 1E-4);
        // 1 0 0
        gini.updateFrequency(1, -1);
        Assertions.assertEquals(1.0, gini.getValue(), 1E-4);
        // 0 0 0
        gini.updateFrequency(0, -1);
        Assertions.assertTrue(Double.isNaN(gini.getValue()));

        gini.updateFrequency(0, 1);
        Assertions.assertEquals(1.0, gini.getValue());

        gini.reset();
        Assertions.assertTrue(Double.isNaN(gini.getValue()));
    }

    @Test
    public void severalSteps()
    {
        Map<Integer, Long> frequencies = new HashMap<>();
        frequencies.put(0, 2L);
        frequencies.put(1, 2L);
        frequencies.put(2, 3L);
        frequencies.put(3, 4L);

        // 2 2 3 4
        GiniIndex gini = new GiniIndex(4, frequencies);
        Assertions.assertEquals(7.0 / 33.0, gini.getValue(), 1E-4);
        // 4 2 3 4
        gini.updateFrequency(0, 2);
        Assertions.assertEquals(7.0 / 39.0, gini.getValue(), 1E-4);
        // 2 2 3 4
        gini.updateFrequency(0, -2);
        Assertions.assertEquals(7.0 / 33.0, gini.getValue(), 1E-4);
        // 5 2 3 4
        gini.updateFrequency(0, 3);
        Assertions.assertEquals(5.0 / 21.0, gini.getValue(), 1E-4);
        // 2 2 3 4
        gini.updateFrequency(0, -3);
        Assertions.assertEquals(7.0 / 33.0, gini.getValue(), 1E-4);
        // 2 2 3 0
        gini.updateFrequency(3, -4);
        Assertions.assertEquals(3.0 / 7.0, gini.getValue(), 1E-4);
        // 2 2 3 5
        gini.updateFrequency(3, 5);
        Assertions.assertEquals(5.0 / 18.0, gini.getValue(), 1E-4);
        // 5 2 3 5
        gini.updateFrequency(0, 3);
        Assertions.assertEquals(11.0 / 45.0, gini.getValue(), 1E-4);
        // 1 2 3 5
        gini.updateFrequency(0, -4);
        Assertions.assertEquals(13.0 / 33.0, gini.getValue(), 1E-4);
        // 5 2 3 5
        gini.updateFrequency(0, 4);
        Assertions.assertEquals(11.0 / 45.0, gini.getValue(), 1E-4);

        gini.reset(frequencies);
        Assertions.assertEquals(7.0 / 33.0, gini.getValue(), 1E-4);

        gini.reset();
        Assertions.assertTrue(Double.isNaN(gini.getValue()));

        gini.updateFrequency(0, 1000);
        Assertions.assertEquals(1.0, gini.getValue(), 1E-4);

        gini.updateFrequency(0, -1000);
        Assertions.assertTrue(Double.isNaN(gini.getValue()));
    }

}
