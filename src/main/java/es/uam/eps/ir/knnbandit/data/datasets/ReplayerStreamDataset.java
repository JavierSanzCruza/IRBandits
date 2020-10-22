package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.datasets.reader.LogRegister;
import es.uam.eps.ir.knnbandit.data.datasets.reader.SimpleStreamDatasetReader;
import es.uam.eps.ir.knnbandit.data.datasets.reader.StreamDatasetReader;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.ranksys.formats.index.ItemsReader;
import org.ranksys.formats.index.UsersReader;
import org.ranksys.formats.parsing.Parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ReplayerStreamDataset<U,I> implements StreamDataset<U,I>
{
    private final StreamDatasetReader<U,I> datasetReader;
    private final FastUserIndex<U> uIndex;
    private final FastItemIndex<I> iIndex;
    private LogRegister<U,I> currentReg;

    public ReplayerStreamDataset(FastUserIndex<U> uIndex, FastItemIndex<I> iIndex, StreamDatasetReader<U,I> reader)
    {
        this.datasetReader = reader;
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.currentReg = null;

    }

    @Override
    public void restart() throws IOException
    {
        datasetReader.initialize();
    }

    @Override
    public void advance() throws IOException
    {
        this.currentReg = this.datasetReader.readRegister();
    }

    @Override
    public boolean hasEnded()
    {
        return this.datasetReader.hasEnded();
    }

    @Override
    public U getCurrentUser()
    {
        if(this.currentReg == null) return null;
        return currentReg.getUser();
    }

    @Override
    public List<I> getCandidateItems()
    {
        if(this.currentReg == null) return null;
        return new ArrayList<>(currentReg.getCandidateItems());
    }

    @Override
    public I getFeaturedItem()
    {
        if(this.currentReg == null) return null;
        return currentReg.getFeaturedItem();
    }

    @Override
    public double getFeaturedItemRating()
    {
        if(this.currentReg == null) return Double.NaN;
        return currentReg.getRating();
    }

    @Override
    public int getCurrentUidx()
    {
        if(this.currentReg == null) return -1;
        return this.user2uidx(currentReg.getUser());
    }

    @Override
    public IntList getCandidateIidx()
    {
        IntList list = new IntArrayList();
        if(this.currentReg == null) return null;
        currentReg.getCandidateItems().forEach(item -> list.add(this.item2iidx(item)));
        return list;
    }

    @Override
    public int getFeaturedIidx()
    {
        if(this.currentReg == null) return -1;
        return this.item2iidx(currentReg.getFeaturedItem());
    }

    @Override
    public int item2iidx(I i)
    {
        return this.iIndex.item2iidx(i);
    }

    @Override
    public I iidx2item(int iidx)
    {
        return this.iIndex.iidx2item(iidx);
    }

    @Override
    public int numItems()
    {
        return this.iIndex.numItems();
    }

    @Override
    public int user2uidx(U u)
    {
        return this.uIndex.numUsers();
    }

    @Override
    public U uidx2user(int uidx)
    {
        return this.uIndex.uidx2user(uidx);
    }

    @Override
    public int numUsers()
    {
        return this.uIndex.numUsers();
    }

    @Override
    public int addItem(I i)
    {
        return 0;
    }

    @Override
    public int addUser(U u)
    {
        return 0;
    }

    /**
     * Loads a stream dataset for the replayer evaluation algorithm.
     * @param input the file containing the data.
     * @param userIndex the file containing user information.
     * @param itemIndex the file containing item information.
     * @param separator the separator for the file.
     * @param uParser the user parser.
     * @param iParser the item parser
     * @return the stream dataset.
     * @throws IOException if something fails while reading the file.
     */
    public static <U,I> ReplayerStreamDataset<U,I> load(String input, String userIndex, String itemIndex, String separator, Parser<U> uParser, Parser<I> iParser) throws IOException
    {
        // First, we read the user index
        FastUpdateableUserIndex<U> uIndex = SimpleFastUpdateableUserIndex.load(UsersReader.read(userIndex, uParser));
        FastUpdateableItemIndex<I> iIndex = SimpleFastUpdateableItemIndex.load(ItemsReader.read(itemIndex, iParser));
        StreamDatasetReader<U,I> streamReader = new SimpleStreamDatasetReader<>(input, uParser, iParser, separator);
        return new ReplayerStreamDataset<>(uIndex, iIndex, streamReader);


    }
}
