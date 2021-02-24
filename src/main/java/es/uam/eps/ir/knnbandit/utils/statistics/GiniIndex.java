package es.uam.eps.ir.knnbandit.utils.statistics;

import it.unimi.dsi.fastutil.ints.Int2LongMap;
import it.unimi.dsi.fastutil.ints.Int2LongOpenHashMap;
import it.unimi.dsi.fastutil.longs.*;

import java.util.Collections;
import java.util.Map;

/**
 * Class for computing and updating the Gini index.
 */
public class GiniIndex
{
    /**
     * For each item in the collection, stores the number of times it has been recommended.
     */
    private final Int2LongMap frequencies;
    /**
     * The minimum indexes for the different possible frequency values.
     */
    private final Long2IntMap mins;
    /**
     * The maximum indexes for the different possible frequencies values.
     */
    private final Long2IntMap maxs;
    /**
     * The total number of items.
     */
    private final int numElements;

    /**
     * List containing the values for the elements.
     */
    private final LongList values;
    /**
     * The sum of the frequencies of all items.
     */
    private double freqSum;
    /**
     * The main term of the Gini index.
     */
    private double numSum;


    /**
     * Constructor. Assumes that the initial values for frequencies are equal to 0.
     *
     * @param numElements the number of elements to consider.
     */
    public GiniIndex(int numElements)
    {
        // initialize:
        this.numElements = numElements;

        // Initialize the sums to zero:
        // The sum of all the frequencies.
        this.freqSum = 0.0;
        // The current value for the numerator of the Gini coefficient.
        this.numSum = 0.0;

        // Initialize the minimums and the maximums. As we do not have any additional
        // information, we only add the 0:
        this.mins = new Long2IntOpenHashMap();
        this.maxs = new Long2IntOpenHashMap();

        this.mins.put(0L, 1);
        this.maxs.put(0L, numElements);

        // Initialize the frequency values:
        this.frequencies = new Int2LongOpenHashMap();
        this.frequencies.defaultReturnValue(0L);

        // Initialize the list of values. Only 0L value exists.
        this.values = new LongArrayList();
        values.add(0L);
    }

    /**
     * Constructor. Uses an initial setting for the different values.
     *
     * @param numElements total number of items.
     * @param frequencies frequencies for the different item values.
     */
    public GiniIndex(int numElements, Map<Integer, Long> frequencies)
    {
        // Initialize the different variables.
        this.numElements = numElements;
        this.freqSum = 0.0;
        this.numSum = 0.0;

        // Initialize the maps for minimums and maximums
        this.mins = new Long2IntOpenHashMap();
        this.maxs = new Long2IntOpenHashMap();

        // Initialize the frequencies of the different items.
        this.frequencies = new Int2LongOpenHashMap();

        // Initialize the list of values.
        this.values = new LongArrayList();

        // Fill the previously created structures with data.
        this.fillValues(frequencies);
    }

    /**
     * Obtains the current value of the Gini index.
     *
     * @return the current value: a number between 0 and 1 representing the proper value of the index,
     * NaN if the frequencies are all equal to zero, or there is less than one element in the collection.
     */
    public double getValue()
    {
        if (this.numElements <= 1)
        {
            return Double.NaN;
        }
        else if (this.freqSum == 0.0)
        {
            return Double.NaN;
        }
        else
        {
            return this.numSum / ((this.numElements - 1.0) * this.freqSum);
        }
    }

    /**
     * Updates the different variables for the Gini index, considering that the
     *
     * @param idx       the index of the element to increase.
     * @param increment how much the frequency varies.
     * @return true if everything is OK, false otherwise
     */
    public boolean updateFrequency(int idx, int increment)
    {
        // ERROR CASE: the value of the index is invalid.
        if (idx < 0 || idx >= this.numElements)
        {
            return false;
        }

        if(this.freqSum + increment >= 0)
        {
            this.freqSum += increment;
            if (increment > 0) // Increment the frequency of the element.
            {
                return this.updateFrequencyIncrease(idx, increment);
            }
            else if (increment < 0) // Decrease the frequency of the element
            {
                return this.updateFrequencyDecrease(idx, -increment);
            }
            // In case increment == 0
            return true;
        }

        return false;
    }

    /**
     * Updates the value for the Gini index if the frequency of an element decreases.
     * A value cannot descend lower than 0.
     *
     * @param idx       the identifier of the element.
     * @param decrement the decrement.
     * @return true if the decrease could be computed, false otherwise.
     */
    private boolean updateFrequencyDecrease(int idx, int decrement)
    {
        // Update the value of numSum.
        // First, get the frequency of item iidx.
        long oldFreq = this.frequencies.get(idx);
        long newFreq = oldFreq - decrement;
        if(newFreq < 0) return false;
        this.frequencies.put(idx, newFreq);

        // Obtain the minimum and maximum indexes for the old value.
        int minOldFreq = this.mins.get(oldFreq);
        int maxOldFreq = this.maxs.get(oldFreq);
        int oldIndex = Collections.binarySearch(this.values, oldFreq);
        boolean delete = (minOldFreq == maxOldFreq);

        // Obtain the minimum and maximum indexes for the new value.
        int minNewFreq;
        int maxNewFreq;
        int newIndex = Collections.binarySearch(this.values, newFreq);
        boolean add = (newIndex < 0);

        if (add)
        {
            int insertion = -newIndex;
            newIndex = insertion - 1;
            minNewFreq = this.mins.get(this.values.get(newIndex));
            maxNewFreq = minNewFreq;
        }
        else
        {
            minNewFreq = this.mins.get(newFreq);
            maxNewFreq = this.maxs.get(newFreq);
        }

        // Now, compute:
        double increase = 0.0;
        // j = freq(idx)
        increase += (this.numElements + 1 - 2 * minOldFreq) * oldFreq;
        // j = freq(idx) + increment
        increase += (add ? 2 * minNewFreq - this.numElements - 1 : 2 * maxNewFreq - this.numElements + 1) * newFreq;

        // For the frequencies between freq(idx) and freq(idx) + increment (not included)
        for (int i = (add ? newIndex : newIndex + 1); i < oldIndex; ++i)
        {
            long freq = this.values.get(i);
            int min = this.mins.get(freq);
            int max = this.maxs.get(freq);

            this.mins.put(freq, min + 1);
            this.maxs.put(freq, max + 1);

            increase += (2 * max - 2 * min + 2) * freq;
        }

        this.numSum += increase;

        if (delete)
        {
            this.values.remove(oldIndex);
            this.mins.remove(oldFreq);
            this.maxs.remove(oldFreq);
        }
        else
        {
            this.mins.put(oldFreq, minOldFreq + 1);
        }

        if (add)
        {
            this.values.add(newIndex, newFreq);
            this.mins.put(newFreq, minNewFreq);
            this.maxs.put(newFreq, maxNewFreq);
        }
        else
        {
            this.maxs.put(newFreq, maxNewFreq + 1);
        }

        return true;
    }

    /**
     * Updates the value of the Gini index when the frequency of an element is increased.
     *
     * @param idx       index of the element to update.
     * @param increment the amount that frequency for the element is increased.
     * @return true if everything is OK, false otherwise.
     */
    private boolean updateFrequencyIncrease(int idx, int increment)
    {
        // Update the value of numSum.
        // First, get the frequency of item iidx.
        long oldFreq = this.frequencies.get(idx);
        long newFreq = oldFreq + increment;
        this.frequencies.put(idx, newFreq);

        // Obtain the minimum and maximum indexes for the old value.
        int minOldFreq = this.mins.get(oldFreq);
        int maxOldFreq = this.maxs.get(oldFreq);
        int oldIndex = Collections.binarySearch(this.values, oldFreq);
        boolean delete = (minOldFreq == maxOldFreq);

        // Obtain the minimum and maximum indexes for the new value.
        int minNewFreq;
        int maxNewFreq;
        int newIndex = Collections.binarySearch(this.values, newFreq);
        boolean add = (newIndex < 0);

        if (add)
        {
            int insertion = -newIndex - 2;
            newIndex = insertion + 1;
            minNewFreq = this.maxs.get(this.values.get(insertion));
            maxNewFreq = minNewFreq;
        }
        else
        {
            minNewFreq = this.mins.get(newFreq);
            maxNewFreq = this.maxs.get(newFreq);
        }

        // Now, compute:
        double increase = 0.0;
        // j = freq(idx)
        increase += (this.numElements + 1 - 2 * maxOldFreq) * oldFreq;
        // j = freq(idx) + increment
        increase += (add ? 2 * minNewFreq - this.numElements - 1 : 2 * minNewFreq - this.numElements - 3) * newFreq;

        // For the frequencies between freq(idx) and freq(idx) + increment (not included)
        for (int i = oldIndex + 1; i < newIndex; ++i)
        {
            long freq = this.values.get(i);
            int min = this.mins.get(freq);
            int max = this.maxs.get(freq);

            this.mins.put(freq, min - 1);
            this.maxs.put(freq, max - 1);

            increase += (2 * min - 2 * max - 2) * freq;
        }

        this.numSum += increase;

        if (add)
        {
            this.values.add(newIndex, newFreq);
            this.mins.put(newFreq, minNewFreq);
            this.maxs.put(newFreq, maxNewFreq);
        }
        else
        {
            this.mins.put(newFreq, minNewFreq - 1);
        }


        if (delete)
        {
            this.values.remove(oldIndex);
            this.mins.remove(oldFreq);
            this.maxs.remove(oldFreq);
        }
        else
        {
            this.maxs.put(oldFreq, maxOldFreq - 1);
        }

        return true;
    }

    /**
     * Given a relation of items and frequencies, updates the values of the
     * items indicated in the relation. This method supposes that the CumulativeGini
     * object is empty.
     *
     * @param frequencies the relation between elements and its frequencies.
     */
    private void fillValues(Map<Integer, Long> frequencies)
    {
        Long2IntSortedMap counter = new Long2IntAVLTreeMap();
        counter.defaultReturnValue(0);
        frequencies.forEach((key, value1) ->
        {
            int item = key;
            long value = value1;

            if (item >= 0 && item < numElements)
            {
                this.frequencies.put(item, value);
                freqSum += value + 0.0;
                int count = counter.getOrDefault(value, counter.defaultReturnValue()) + 1;
                counter.put(value, count);
            }
        });

        int numElemsWithoutFreq = numElements - frequencies.size();
        if (numElemsWithoutFreq > 0)
        {
            this.mins.put(0L, 1);
            this.maxs.put(0L, numElemsWithoutFreq);
            this.values.add(0L);
        }

        int currentIndex = numElemsWithoutFreq;
        for (long freq : counter.keySet())
        {
            this.values.add(freq);
            int count = counter.get(freq);
            int min = currentIndex + 1;
            int max = currentIndex + count;

            this.mins.put(freq, currentIndex + 1);
            currentIndex += counter.get(freq);
            this.maxs.put(freq, currentIndex);
            numSum += freq * ((max - min + 1) * (max + min - numElements - 1));
        }
    }

    /**
     * Resets the metric to the state with no initial information.
     */
    public void reset()
    {
        this.maxs.clear();
        this.mins.clear();
        this.values.clear();
        this.numSum = 0.0;
        this.freqSum = 0.0;
        this.frequencies.clear();

        this.maxs.put(0L, this.numElements);
        this.mins.put(0L, 1);
        this.values.add(0);
    }

    /**
     * Resets the metric to an initial state.
     *
     * @param frequencies the initial frequencies.
     */
    public void reset(Map<Integer, Long> frequencies)
    {
        this.maxs.clear();
        this.mins.clear();
        this.values.clear();
        this.numSum = 0.0;
        this.freqSum = 0.0;
        this.frequencies.clear();

        this.fillValues(frequencies);
    }
}
