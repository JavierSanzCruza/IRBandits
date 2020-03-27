package es.uam.eps.ir.knnbandit.recommendation;

public enum KnowledgeDataUse
{
    ONLYKNOWN, ONLYUNKNOWN, ALL;

    /**
     * Obtains the KnowledgeDataUse value from a string.
     *
     * @param string known for ONLYKNOWN, unknown for ONLYUNKNOWN, all else for ALL.
     * @return the KnowledgeDataUse value (ALL by default).
     */
    public static KnowledgeDataUse fromString(String string)
    {
        String aux = string.toLowerCase();
        switch (aux)
        {
            case "known":
                return ONLYKNOWN;
            case "unknown":
                return ONLYUNKNOWN;
            case "all":
            default:
                return ALL;
        }
    }
}
