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
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.core.util.tuples.Tuple2id;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Class for reading a recommendation file in text mode.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class TextReader implements Reader
{
    /**
     * A writer, for printing the results into a file.
     */
    private BufferedReader br;

    /**
     * The next line to process (if any)
     */
    private String nextLine = null;

    /**
     * The size of a register.
     */
    int headerSize = 0;

    @Override
    public void initialize(InputStream inputStream) throws IOException
    {
        if(this.br != null) throw new IOException("ERROR: there is a file currently open");
        this.br = new BufferedReader(new InputStreamReader(inputStream));
    }

    @Override
    public void initialize(String filename) throws IOException
    {
        if(this.br != null) throw new IOException("ERROR: there is a file currently open");
        this.br = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
    }

    @Override
    public List<String> readHeader() throws IOException
    {
        String line = br.readLine();
        String[] split = line.split("\t");
        headerSize = split.length;
        return new ArrayList<>(Arrays.asList(split));
    }

    @Override
    public Tuple3<Integer, FastRecommendation, Long> readIteration() throws IOException
    {
        if(nextLine == null)
        {
            nextLine = br.readLine();
            if(nextLine == null)
            {
                return null;
            }
        }

        String[] split = nextLine.split("\t");
        int currentIter = Parsers.ip.parse(split[0]);
        int uidx = Parsers.ip.parse(split[1]);
        int iidx = Parsers.ip.parse(split[2]);
        long time = Parsers.lp.parse(split[split.length-1]);
        IntList recs = new IntArrayList();
        recs.add(iidx);

        boolean stop = false;
        boolean store = true;

        while(!stop)
        {
            nextLine = br.readLine();
            if(nextLine != null)
            {
                split = nextLine.split("\t");
                int iter = Parsers.ip.parse(split[0]);
                if(iter != currentIter)
                {
                    stop = true;
                }

                if(split.length < headerSize)
                {
                    nextLine = null;
                    store = stop;
                }
                else if(!stop)
                {
                    iidx = Parsers.ip.parse(split[2]);
                    recs.add(iidx);
                }
            }
            else
            {
                stop = true;
            }
        }

        if(store)
        {
            List<Tuple2id> rec = new ArrayList<>();
            for (int i = 0; i < recs.size(); ++i)
            {
                rec.add(new Tuple2id(recs.getInt(i), (recs.size() - i + 0.0) / (recs.size() + 0.0)));
            }

            return new Tuple3<>(currentIter, new FastRecommendation(uidx, rec), time);
        }

        return null;
    }

    @Override
    public void close() throws IOException
    {
        if (this.br != null)
        {
            this.br.close();
        }
        this.br = null;
    }

    @Override
    public List<Pair<Integer>> readFile(String filename) throws IOException
    {
        this.initialize(filename);
        List<String> header = this.readHeader();
        List<Pair<Integer>> list = new ArrayList<>();
        String line;
        while((line = br.readLine()) != null)
        {
            String[] split = line.split("\t");
            if(split.length < header.size())
            {
                break;
            }

            list.add(new Pair<>(Parsers.ip.parse(split[1]), Parsers.ip.parse(split[2])));
        }
        this.close();
        return list;
    }

    @Override
    public List<Pair<Integer>> readFile(InputStream stream) throws IOException
    {
        this.initialize(stream);
        List<String> header = this.readHeader();
        List<Pair<Integer>> list = new ArrayList<>();
        String line;
        while((line = br.readLine()) != null)
        {
            String[] split = line.split("\t");
            if(split.length < header.size())
            {
                break;
            }

            list.add(new Pair<>(Parsers.ip.parse(split[1]), Parsers.ip.parse(split[2])));
        }
        this.close();
        return list;
    }
}