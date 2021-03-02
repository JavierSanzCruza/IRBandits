/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.data.datasets.reader;

import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.HashSet;

/**
 * Reader for a stream dataset.
 * @param <U> type of the users.
 * @param <I> type of the items.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public abstract class StreamDatasetReader<U,I>
{
    /**
     * The file containing the dataset.
     */
    protected final String file;
    /**
     * True if the dataset has been fully processed.
     */
    protected boolean finished;
    /**
     * The buffered reader.
     */
    private BufferedReader br;
    /**
     * User parser.
     */
    protected final Parser<U> uParser;
    /**
     * Item parser.
     */
    protected final Parser<I> iParser;
    /**
     * Separator between fields in a register.
     */
    protected final String separator;

    /**
     * Constructor.
     * @param file      the route to the dataset file.
     * @param uParser   a parser for reading the users.
     * @param iParser   a parser for reading the items.
     * @param separator the separator between the fields in a register in the dataset.
     */
    public StreamDatasetReader(String file, Parser<U> uParser, Parser<I> iParser, String separator)
    {
        this.file = file;
        this.uParser = uParser;
        this.iParser = iParser;
        this.separator = separator;
        this.finished = true;
    }

    /**
     * Initializes the reader.
     * @throws IOException if something fails while reading the stream dataset.
     */
    public void initialize() throws IOException
    {
        if(br != null) br.close();
        br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        this.finished = false;
    }

    /**
     * Reads an individual register.
     * @return the individual register.
     * @throws IOException if something fails while reading the register.
     */
    public LogRegister<U,I> readRegister() throws IOException
    {
        LogRegister<U,I> register = null;
        if(!finished)
        {
            String line = br.readLine();
            if (line == null)
            {
                br.close();
                br = null;
                finished = true;
            }
            else
            {
                register = processRegister(line);
                // Process the register:
                String[] split = line.split(separator);
                U u = uParser.parse(split[0]);
                I i = iParser.parse(split[1]);
                double value = Parsers.dp.parse(split[2]);
                Collection<I> candidates = new HashSet<>();
                for (int j = 3; j < split.length; ++j)
                {
                    candidates.add(iParser.parse(split[j]));
                }
                if (!candidates.contains(i)) candidates.add(i);
            }
        }
        return register;
    }

    /**
     * Given a line from the dataset, builds a register.
     * @param line the line.
     * @return the register.
     */
    protected abstract LogRegister<U,I> processRegister(String line);

    /**
     * Checks whether we have read the whole dataset or not.
     * @return true if we have, false otherwise.
     */
    public boolean hasEnded()
    {
        return this.finished;
    }

    /**
     * Closes the reader.
     * @throws IOException if something fails while closing the reader.
     */
    public void close() throws IOException
    {
        if(this.br != null)
        {
            this.br.close();
            br = null;
            this.finished = true;
        }
    }
}
