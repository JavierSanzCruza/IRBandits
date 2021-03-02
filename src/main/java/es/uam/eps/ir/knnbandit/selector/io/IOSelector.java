/*
 *  Copyright (C) 2020 Information Retrieval Group at Universidad Autónoma
 *  de Madrid, http://ir.ii.uam.es
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package es.uam.eps.ir.knnbandit.selector.io;

import es.uam.eps.ir.knnbandit.io.*;
import es.uam.eps.ir.knnbandit.io.Reader;
import es.uam.eps.ir.knnbandit.io.Writer;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Class for selecting a given reader/writer.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 * @author Pablo Castells (pablo.castells@uam.es)
 */
public class IOSelector
{
    /**
     * The type of input / output.
     */
    private final IOType type;

    /**
     * True if the files have to be compressed (using GZIP)
     */
    private final boolean gzipped;

    /**
     * Constructor.
     * @param type the selected type.
     */
    public IOSelector(IOType type, boolean gzipped)
    {
        this.type = type;
        this.gzipped = gzipped;
    }

    /**
     * Obtains a new reader.
     * @return the new reader if the type is available, null otherwise.
     */
    public Reader getReader()
    {
        switch (this.type)
        {
            case BINARY:
                return new BinaryReader();
            case TEXT:
                return new TextReader();
            case ERROR:
            default:
                return null;
        }
    }

    /**
     * Obtains a new writer.
     * @return the new writer if the type is available, null otherwise.
     */
    public Writer getWriter()
    {
        switch (this.type)
        {
            case BINARY:
                return new BinaryWriter();
            case TEXT:
                return new TextWriter();
            case ERROR:
            default:
                return null;
        }
    }

    /**
     * Returns an input stream for reading the files.
     * @param filename the name of the file.
     * @return a stream for reading the corresponding recommendation files.
     * @throws IOException if something fails while creating the stream.
     */
    public InputStream getInputStream(String filename) throws IOException
    {
        return gzipped ? new GZIPInputStream(new FileInputStream(filename)) : new FileInputStream(filename);
    }

    /**
     * Returns an output stream for writing in the files.
     * @param filename the name of the file.
     * @return a stream for writing the corresponding recommendation files.
     * @throws IOException if something fails while creating the stream.
     */
    public OutputStream getOutputStream(String filename) throws IOException
    {
        return gzipped ? new GZIPOutputStream(new FileOutputStream(filename)) : new FileOutputStream(filename);
    }

    /**
     * Obtains whether the input-output files have to be compressed or not.
     * @return true if the input-output files have to be compressed, false otherwise.
     */
    public boolean isCompressed()
    {
        return gzipped;
    }
}