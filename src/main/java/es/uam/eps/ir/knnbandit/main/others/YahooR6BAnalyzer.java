package es.uam.eps.ir.knnbandit.main.others;

import es.uam.eps.ir.knnbandit.data.preference.updateable.fast.AdditiveRatingFastUpdateablePreferenceData;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.updateable.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.utils.Pair;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import org.ranksys.formats.parsing.Parsers;

import java.io.*;
import java.util.*;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class YahooR6BAnalyzer
{
    public static void main(String[] args) throws IOException
    {
        String input = args[0];

        ZipFile file = new ZipFile(input);
        Enumeration<? extends ZipEntry> entries = file.entries();

        FastUpdateableUserIndex<String> uIndex = new SimpleFastUpdateableUserIndex<>();
        FastUpdateableItemIndex<String> iIndex = new SimpleFastUpdateableItemIndex<>();
        AdditiveRatingFastUpdateablePreferenceData<String, String> numTimes = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);
        AdditiveRatingFastUpdateablePreferenceData<String, String> numPos = AdditiveRatingFastUpdateablePreferenceData.load(Stream.empty(), uIndex, iIndex);

        while(entries.hasMoreElements())
        {
            ZipEntry entry = entries.nextElement();
            GZIPInputStream inputStream = new GZIPInputStream(file.getInputStream(entry));
            try(BufferedReader br = new BufferedReader(new InputStreamReader(inputStream)))
            {
                br.lines().forEach(line ->
                {
                    String[] split = line.split("\\s+");
                    String item = split[1];
                    numTimes.addItem(item);
                    numPos.addItem(item);
                    double rating = Parsers.dp.parse(split[2]);
                    boolean isCurrentUser = false;
                    boolean isEmptyUser = true;
                    char[] user = new char[135];
                    for(int i = 3; i < split.length; ++i)
                    {
                        if(split[i].equals( "|user"))
                        {
                            isCurrentUser = true;
                        }
                        else if(split[i].startsWith("|"))
                        {
                            isCurrentUser = false;
                            String itemId = split[i].substring(1);
                            numTimes.addItem(item);
                            numPos.addItem(item);
                        }

                        if(isCurrentUser)
                        {
                            int index = Parsers.ip.parse(split[i]);
                            if(index > 1)
                            {
                                isEmptyUser = false;
                                user[index-2] = '1';
                            }
                        }
                    }

                    if(!isEmptyUser)
                    {
                        String userId = new String(user);
                        numTimes.addUser(userId);
                        numPos.addUser(userId);

                        numTimes.update(userId, item, 1.0);
                        numPos.update(userId, item, rating);
                    }

                });
            }

        }

        // now, print everything:
        // STEP 1: the user index file
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + "users.txt")));
            BufferedWriter bwUserData = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + "userdata.txt"))))
        {
            bwUserData.write("userId\tnumRatings\tnumRel\tnumRep\tnumRepRel\n");
            uIndex.getAllUsers().forEach(user ->
            {
                try
                {
                    bw.write(user);
                    bwUserData.write(user);
                    if(numTimes.numItems(user) > 0)
                    {
                        Pair<Double> pair1 = numTimes.getUserPreferences(user).map(pref -> new Pair<>(pref.v2, pref.v2 - 1)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        Pair<Double> pair2 = numPos.getUserPreferences(user).map(pref -> new Pair<>(pref.v2, pref.v2 > 1 ? pref.v2 - 1 : 0)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        bw.write("\t" + pair1.v1() + "\t" + pair2.v1() + "\t" + pair1.v2() + "\t" + pair2.v2());
                    }
                    else
                    {
                        bwUserData.write("\t0\t0\t0\n");
                    }
                }
                catch(IOException ioe)
                {
                    System.err.println("Something wrong occurred");
                }
            });

        }

        // STEP 2: the item index files
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + "items.txt")));
            BufferedWriter bwUserData = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + "itemdata.txt"))))
        {
            bwUserData.write("itemId\tnumRatings\tnumRel\tnumRep\tnumRepRel\n");
            iIndex.getAllItems().forEach(item ->
            {
                try
                {
                    bw.write(item);
                    bwUserData.write(item);
                    if(numTimes.numItems(item) > 0)
                    {
                        Pair<Double> pair1 = numTimes.getItemPreferences(item).map(pref -> new Pair<>(pref.v2, pref.v2 - 1)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        Pair<Double> pair2 = numPos.getItemPreferences(item).map(pref -> new Pair<>(pref.v2, pref.v2 > 1 ? pref.v2 - 1 : 0)).reduce(new Pair<>(0.0,0.0), (x,y) -> new Pair<>(x.v1()+y.v1(), x.v2()+y.v2()));
                        bw.write("\t" + pair1.v1() + "\t" + pair2.v1() + "\t" + pair1.v2() + "\t" + pair2.v2());
                    }
                    else
                    {
                        bwUserData.write("\t0\t0\t0\n");
                    }
                }
                catch(IOException ioe)
                {
                    System.err.println("Something wrong occurred");
                }
            });
        }

        // STEP 3: the ratings file
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1] + "ratings.txt"))))
        {
            bw.write("userId\titemId\tnumPos\tnumTimes\n");
            numTimes.getUsersWithPreferences().forEach(user ->
                numTimes.getUserPreferences(user).forEach(item ->
                {
                    try
                    {
                        String i = item.v1();
                        Optional<? extends IdPref<String>> opt = numPos.getPreference(user, i);
                        if (opt.isPresent())
                        {
                            bw.write(user + "\t" + i + "\t" + opt.get() + "\t" + item.v2());
                        }
                        else
                        {
                            bw.write(user + "\t" + i + "\t" + 0.0 + "\t" + item.v2());

                        }
                    }
                    catch(IOException ioe)
                    {
                        System.err.println("Something wrong occurred");
                    }
                })
            );
        }
    }
}
