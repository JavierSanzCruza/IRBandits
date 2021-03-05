/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.io;

import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Class that reads a recommendation from a binary file.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class BinaryReader implements Reader
{
    /**
     * The input stream.
     */
    private DataInputStream inputStream = null;

    @Override
    public void initialize(String filename) throws IOException
    {
        if(this.inputStream != null) throw new IOException("ERROR: there is a file currently open");
        this.inputStream = new DataInputStream(new FileInputStream(filename));
    }

    @Override
    public void initialize(InputStream inputStream) throws IOException
    {
        if(this.inputStream != null) throw new IOException("ERROR: there is a file currently open");
        this.inputStream = new DataInputStream(inputStream);
    }

    @Override
    public Tuple3<Integer, FastRecommendation, Long> readIteration() throws IOException
    {
        try
        {
            int numIter = inputStream.readInt();
            int uidx = inputStream.readInt();
            int numItems = inputStream.readInt();
            List<Tuple2id> list = new ArrayList<>();
            long time;

            for(int i = 0; i < numItems; ++i)
            {
                list.add(new Tuple2id(inputStream.readInt(), (numItems-i+0.0)/(numItems)));
            }
            time = inputStream.readLong();
            return new Tuple3<>(numIter, new FastRecommendation(uidx, list), time);
        }
        catch(EOFException eof)
        {
            return null;
        }
    }

    @Override
    public void close() throws IOException
    {
        inputStream.close();
        inputStream = null;
    }

    @Override
    public List<String> readHeader()
    {
        return new ArrayList<>();
    }

    @Override
    public List<Pair<Integer>> readFile(String filename) throws IOException
    {
        this.initialize(filename);
        this.readHeader();
        List<Pair<Integer>> rec = new ArrayList<>();
        Tuple3<Integer, FastRecommendation, Long> indiv;
        while((indiv = this.readIteration()) != null)
        {
            int uidx = indiv.v2.getUidx();
            for(Tuple2id tuple : indiv.v2.getIidxs())
            {
                rec.add(new Pair<>(uidx, tuple.v1));
            }
        }
        this.close();
        return rec;
    }

    @Override
    public List<Pair<Integer>> readFile(InputStream stream) throws IOException
    {
        this.initialize(stream);
        this.readHeader();
        List<Pair<Integer>> rec = new ArrayList<>();
        Tuple3<Integer, FastRecommendation, Long> indiv;
        while((indiv = this.readIteration()) != null)
        {
            int uidx = indiv.v2.getUidx();
            for(Tuple2id tuple : indiv.v2.getIidxs())
            {
                rec.add(new Pair<>(uidx, tuple.v1));
            }
        }
        this.close();
        return rec;
    }
}