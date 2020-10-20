package es.uam.eps.ir.knnbandit.data.datasets;

import es.uam.eps.ir.knnbandit.data.datasets.reader.LogRegister;
import es.uam.eps.ir.knnbandit.data.datasets.reader.StreamDatasetReader;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

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
}
