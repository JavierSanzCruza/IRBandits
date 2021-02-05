# IR bandits
Interactive recommendation library.

## Authors
Information Retrieval Group at Universidad Autónoma de Madrid
- Javier Sanz-Cruzado (javier.sanz-cruzado@uam.es)
- Pablo Castells (pablo.castells@uam.es)

## Software description
This repository contains all the needed classes to reproduce the experiments explained in the paper. The software contains the following packages:
- `es.uam.eps.ir.knnbandit.data`: Classes for handling the ratings by users for items. Extension of the RankSys preference data classes to allow the addition of new users, items and ratings.
- `es.uam.eps.ir.knnbandit.graph`: Classes for handling graph data for contact recommendation.
- `es.uam.eps.ir.knnbandit.io`: Classes for reading/writing the simulation output files.
- `es.uam.eps.ir.knnbandit.main`: Main programs for the execution.
- `es.uam.eps.ir.knnbandit.metrics`: Classes implementing the cumulative metrics used in the experiments.
- `es.uam.eps.ir.knnbandit.recommendation`: Implementation of recommendation algorithms and similarities.
- `es.uam.eps.ir.knnbandit.selector`: Classes for reading the list of algorithms/metrics to execute.
- `es.uam.eps.ir.knnbandit.stats`: Probability distributions.
- `es.uam.eps.ir.knnbandit.utils`: Additional classes, useful for the rest of the program.
- `es.uam.eps.ir.knnbandit.warmup`: Classes for dealing with initial training data.

### Algorithms
The software includes the implementation of several recommendation algorithms.

#### Multi-armed bandits for recommendation
- **kNN bandit:** The main contribution of this paper: we implement our proposed approach by defining a user-based kNN recommender with the appropriate item scoring function, to be used with a stochastic similarity that uses Thompson sampling to estimate the similarities between users [1]. 
- **Interactive matrix factorization:** A probabilistic matrix factorization-based bandit [2]
- **Particle Thompson Sampling matrix factorization:** A probabilistic matrix factorization approach based on Thompson Sampling and particle filtering [3]
- **CLUstering of Bandits (CLUB):** Collaborative filtering version of a clustering-based algorithm [4]
- **COllaborative FIltering BAndit:** Advanced version of the collaborative filtering CLUB algorithm [5]
- **Item-oriented, non-personalized multi-armed bandits:** &epsilon;-greedy [6], &epsilon; t-greedy, UCB1, UCB1-tuned [7], Thompson sampling [8]. They are used as baseline bandit algorithms in the paper.

#### Myopic recommendation algorithms
These approaches are just an incrementally-updateable version of classical recommendation algorithms, used as baselines. The algorithms included in this comparison are:
- **Non-personalized recommendation:** Random recommendation, popularity-based recommendation, average rating.
- **Matrix factorization:** Implicit matrix factorization (iMF) [9], fast iMF [10], pLSA [11].
- **User-based kNN:** Non-normalized implementations of classic user-based cosine kNN [12].

### Metrics
To evaluate and analyze the different algorithms, we implement different metrics:
- **Cumulative Recall:** The proportion of  relevant ratings that have been discovered at a certain point in time.
- **Cumulative Gini:** Measures how imbalanced is the distribution of the number of times each item has been recommended up to some point in time.
- **Cumulative EPC:** Measures how impopular the recommended items are (according to the final distribution).
- **Cumulative ILD:** Measures the diversity between the recommendations of each user.
- **Cumulative Counter:** Counts the number of discovered ratings (either positive or negative).
- **Clickthrough rate:** From all the possible clicks, measures how many of them have been positive.

### Simulation protocols
- **Non-sequential [1]:** This protocol attempts to discover every rating in the test data. It starts from some warmup data, and, each time, selects a user, and recommends him an item (not previously recommended).
- **Non-sequential, with limited item pool [4]:** This protocol, each iteration, takes a user, an item with positive rating, and a bunch of others in the system, and checks whether the algorithm is able to provide the user a positively rated item.
- **Replayer [13]:** Sequential protocol that advances over a large log of interactions.
- **Replayer, with limited item pool [13]:** The same as replayer, but each iteration, only some candidate items are available.
 
## System Requirements
**Java JDK:** 1.8 or above (the software was tested using version 1.8.0_112).

**Maven:** tested with version 3.6.0.

## Installation
In order to install this program, you need to have Maven (https://maven.apache.org) installed on your system. Then, download the files into a directory, and execute the following command:
```
mvn compile assembly::single
```
If you do not want to use Maven, it is still possible to compile the code using any Java compiler. In that case, you will need the following libraries:
- Ranksys version 0.4.3: http://ranksys.org
- Colt version 1.2.0: https://dst.lbl.gov/ACSSoftware/colt
- Google MTJ version 1.0.4: https://github.com/fommil/matrix-toolkits-java
- JSon version 20200518: https://mvnrepository.com/artifact/org.json/json
- Apache Commons Compress version 1.14: https://commons.apache.org/proper/commons-compress/

## Execution

Several programs can be executed with the IR Bandits library. We summarize here the utility of such programs, but the execution details for each of them are included in the project Wiki:

Once you have generated a .jar, you can execute the program. There are two different ways to run this program: one for general recommendation (movies, songs, venues...) and one for contact recommendation in social networks, since the respective evaluation protocols have slight differences between them.

### General recommendation
```
java -jar knnbandit-jar-with-dependencies.jar generalrec algorithmsFile dataFile outputFolder numIter threshold resume binarize
```
where the command line arguments are:
  - `algorithmsFile`: A file indicating which algorithms have to be executed.
  - `dataFile`: The rating data, including one rating per line with the format: `user \t item \t rating`.
  - `outputFolder`: The directory where the output files will be stored.
  - `numIter`: The number of iterations to run for each algorithm. Use value `0` for running until no new items can be recommended.
  - `threshold`: Relevance threshold. Ratings greater than or equal to this value will be considered as relevant.
  - `resume`: Set value to `true` to resume execution following up from the output of a previous execution (if any) or `false` to overwrite and start the interactive recommendation cycle from scratch.
  - `binarize`: Set value to `true` for using binarized rating values (1 for relevant, 0 for non-relevant), `false` to leave rating values as are.
  
For reproducing the exact experiments of the paper, program argument values are:
- `numIter = 500000` for Foursquare-NY, `numIter = 1000000` for Foursquare-Tokyo and `numIter = 3000000` for MovieLens1M.
- `threshold = 1` for Foursquare and `threshold = 4` for MovieLens1M.
- `binarize = true` for all datasets.

### Contact recommendation
```
java -jar knnbandit-jar-with-dependencies.jar contactRec algorithmsFile dataFile outputFolder numIter directed resume notReciprocal
```
where
  - `algorithmsFile`: A file indicating which algorithms have to be executed
  - `dataFile`: The graph data, including one edge per line with the format: `originUser \t destUser \t weight`.
  - `outputFolder`: The directory where the output files will be stored.
  - `numIter`: The number of iterations to run for each algorithm. Use value `0` for running until no new items can be recommended.
  - `directed`: Set value to `true` if the social network is directed, `false` otherwise.
  - `resume`: Set value to `true` to resume execution following up from the output of a previous execution (if any) or `false` to overwrite and start the interactive recommendation cycle from scratch.
  - `notReciprocal`: Set value to `true` if the algorithms should not recommend reciprocal links, `false` otherwise.
  
For reproducing the exact experiments of the paper, the arguments are:
 - `numIter = 5000000`.
 - `directed = true`.
 - `notReciprocal = true`.
 
### Algorithm files
In order to execute different configurations, we include in the `config` folder the optimal configurations for the different datasets we used in the paper. Each row represents the configuration for a single algorithm.

Example:
```
popularity
random
average
ubknn-100
knnbandit-1-1-10
mf-10-fastimf-10-10-20
itembandit-epsilon-0.2-stationary
itembandit-thompson-1-100
```

In the above configuration file example, we choose different algorithms ot be run (each with specific parameter settings): popularity, random recommendation, average rating, myopic user-based kNN, kNN bandit, matrix factorization, &epsilon;-greedy and thompson sampling.

#### Testing different configurations for the kNN bandit
To test the different settings for this algorithm, the format of the line to add in the configuration file is:

   knnbandit-`k`-`alpha`-`beta`
   
where:
 - `k` is the number of neighbors to use.
 - `alpha` is the initial number of hits of the algorithm (high value for optimistic start).
 - `beta` is the initial number of errors of the algorithm (high value for pessimistic start).
  
### Random seed
It is possible to set a random seed for the experiments, so that the selection of users and other random choices are the same when the experiment is repeated. For that purpose, in the output directory, just add a file named `rngseed` (without any file extension) containing the seed in the first line, and set the parameter `resume` to true. We include the random seeds we used in our experiments in the rng-seeds folder. In order to use them, they have to be renamed when added to the output directory.

### Output format
The output of both programs is the same: for each algorithm in the comparison, a file will be created. The name of the file will be the same as the chosen algorithm configuration. Each of the output files has the following format: separated by tabs, the first line contains the header of the file. Then, each row contains the information of a single iteration: the number of the iteration, the selected user, the selected item, the value of the metrics and the time taken to execute the iteration (in ms.)

This is an example of the content format of this file:
```
iter	user	item	recall	gini	time
0	1713	4901	0.0	1.0	27
1	1880	1477	0.0	0.9999838334195551	13
2	1626	56725	0.0	0.9999676668391102	3
3	2002	34539	0.0	0.9999515002586653	3
4	477	5085	0.0	0.9999353336782204	6
5	2012	44312	0.0	0.9999191670977755	5
6	1526	60448	0.0	0.9999030005173306	45
7	528	9392	0.0	0.9998868339368857	46
8	887	2878	0.0	0.9998706673564408	31
9	1313	22947	0.0	0.9998545007759959	56
10	2274	45478	0.0	0.9998383341955509	1
11	1615	7493	0.0	0.999822167615106	2
12	1481	58528	0.0	0.9998060010346611	0
```

## References
1. Sanz-Cruzado, J., Castells, P., López, E. (2019).  A Simple Multi-Armed Nearest-Neighbor Bandit for Interactive Recommendation. In 13th ACM Conference on Recommender Systems (RecSys 2019). Copenhagen, Denmark, September 2019, pp. 358–362.
2. Zhao, X., Zhang, W., Wang, J. (2013). Interactive collaborative filtering. In 22nd ACM international Conference on Information & Knowledge Management (CIKM 2013). San Francisco, California, USA, October 2013, pp. 1411-1420. 
3. Kawale, J., Bui, H., Kveton, B., T., Thanh, L.T., Chawla, S. (2015). Efficient Thompson sampling for online matrix-factorization recommendation. In 29th International Conference on Neural Information Processing Systems (NIPS 2015). Montréal, Canada, December 2015. 
4. Gentile, C., Li, S., Zapella, G. (2014). Online clustering of bandits. In 31st International Conference on Machine Learning (ICML 2014). Beijing, China, June 2014.
5. Li, S., Karatzoglou, A., Gentile, C. (2016). Collaborative Filtering Bandits. In 39th ACM SIGIR conference on Research and Development in Information Retrieval (SIGIR 2016). Pisa, Italy, July 2016, pp 539-548
6. Sutton, R., Barto, A. (2018). Reinforcement Learning: An Introduction, 2nd. Edition.
7. Auer, P., Cesa-Bianchi, N. Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning 47(2-3), pp. 235-256.
8. Chapelle, O., Chang, Y. (2011). An empirical evaluation of Thompson sampling. In 24th International Conference on Neural Information Processing Systems (NIPS 2011). Granada, Spain, December 2011. 
9. Hofmann, T. (2004). Latent semantic models for collaborative filtering. ACM Transactions on Information Systems, 22(1), pp. 89–115
10. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. In 2008 Eighth IEEE International Conference on Data Mining (ICDM 2008). Pisa, Italy, December 2008, pp. 263–272.
11. Pilászy, I., Zibriczky, D., & Tikk, D. (2010). Fast ALS-based matrix factorization for explicit and implicit feedback datasets. In Proceedings of the 4th ACM conference on Recommender systems (Recsys 2010). Barcelona, Spain, September 2010, pp. 71-78.
12. Ning, X., Desrosiers, C., Karypis, G. (2015). A Comprehensive Survey of Neighborhood-Based Recommendation Methods. Recommender Systems Handbook, 2nd Edition, pp. 37-76.