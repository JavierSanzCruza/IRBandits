package es.uam.eps.ir.knnbandit.warmup;

public enum WarmupType
{
    FULL, ONLYRATINGS;

    /**
     * Obtains the warm-up type.
     *
     * @param type the warm-up type selection.
     * @return the type if everything is OK, null otherwise.
     */
    public static WarmupType fromString(String type)
    {
        switch (type.toLowerCase())
        {
            case "onlyratings":
                return ONLYRATINGS;
            case "full":
                return FULL;
            default:
                return null;
        }
    }
}
