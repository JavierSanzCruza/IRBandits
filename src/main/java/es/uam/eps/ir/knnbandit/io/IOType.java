package es.uam.eps.ir.knnbandit.io;

/**
 * Auxiliar class for selecting the type of input/output.
 *
 * @author Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
 */
public enum IOType
{
    BINARY, TEXT, ERROR;

    public static IOType fromString(String str)
    {
        switch (str.toLowerCase())
        {
            case "binary": // for binary files.
                return BINARY;
            case "text": // for text files.
                return TEXT;
            default:
                return ERROR;
        }
    }
}