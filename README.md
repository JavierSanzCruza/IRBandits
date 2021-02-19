![Java CI with Maven](https://github.com/JavierSanzCruza/IRBandits/workflows/Java%20CI%20with%20Maven/badge.svg)
[![GitHub license](https://img.shields.io/badge/license-MPL--2.0-orange)](https://www.mozilla.org/en-US/MPL/)

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

Several programs can be executed in this library. We summarize here the utility of such programs. Execution details for each of them are included in the project Wiki. The different programs have different configurations, depending on the dataset type we want to use. The common execution line is the following:
```
java -jar IRBandits.jar program type-of-dataset program-basic-arguments dataset-related-arguments optional-arguments
```
where the command line arguments are:
 - `program`: The identifier of the program we want to execute. We shall explain them later.
 - `type-of-dataset`: arguments refering to the type of dataset we want to execute. We distinguish four possibilities here:
     - `general` for using a general item recommendation dataset (movies, songs, artists, etc.) like Movielens1M, Netflix, etc. As of now, we distinguish two types:
         - `movielens`: Fields separated by `::`, and uses long values to represent users and items.
         - `foursquare`: Fields separated by `::`, uses long values to represent users, and strings to represent items. 
     - `knowledge` for using a recommendation dataset including information about whether the users knew the items previously to the recommendation (e.g. cm100k). Fields are separated by `::`, as in the `general` case.
     - `contact` for using a contact recommendation dataset, where the "ratings" represent connection between users. Fields are separated by a tab character.
     - `stream` for applying the Replayer evaluation strategy on a dataset.
 - `program-basic-arguments`: program arguments shared by all the dataset types.
 - `dataset-related-arguments`: program arguments related to the specific nature of each dataset. The arguments are the following, and must be introduced in the upcoming order.
     - `general` takes the following arguments:
         - `threshold`: The relevance threshold for the ratings. Ratings with value greater or equal than this shall be considered relevant.
         - `useRatings`: True if the recommenders use the actual bandits, false if the ratings are binarized (if rating is greater or equal than the threshold, it will be assigned value 1, 0 otherwise).
     - `knowledge` takes the arguments for the `general` case, and adds the following one:
         - `dataUse`: Specifies which ratings we use for updating the recommendation approaches: `known` for only using the items the users already knew about before the recommendation, `unknown` for using only those items the target user did not know about, and `all` to use all of them.
     - `contact` takes the following arguments:
         - `directed`: True if the underlying social network is directed, false otherwise.
         - `notReciprocal`: in directed networks, if this value is true, it does not allow to recommend reciprocal edges to those discovered in the network. Note: if the edge (u,v) does not exist, u can still be recommended to v.
     - `stream` uses the following arguments:
         - `threshold`: The relevance threshold for the ratings.  Ratings with value greater or equal than this shall be considered relevant.
         - `userIndex`: In order to read the dataset just once per execution, these programs receive a file containing the users in the dataset (one per line).
         - `itemIndex`: Similarly to the user index, this argument points to a file containing the items in the dataset (one per line).
 - `optional-arguments`: additional arguments that the user might (or might not) introduce for the different programs.

Next, we detail the arguments and utilities of the different programs:

### Validation
Given no warmup, this program executes validation to search for the optimal parameters for a recommendation algorithm. It is executed as:
```
java -jar IRBandits.jar valid type-of-dataset algorithms input output end-condition resume dataset-related-arguments (-k times)
```
where the command line arguments are:
   - `type-of-dataset`: see the earlier type of dataset configuration.
   - `algorithms`: a JSON file containing the possible hyperparameter configurations of algorithms to consider.
   - `input`: file containing the dataset.
   - `output`: the directory in which to store the output.
   - `end-condition`: the end condition for the recommendation. Depending on its value, several possibilities:
       - `end-condition = 0.0`: ends when no user can be recommended anything.
       - `0.0 < end-condition < 1.0`: ends when it has discovered a fraction of the positive ratings equal to `end-condition`.
       - `end-condition > 1.0`: executes it for a number of iterations equal to `end-condition`
   - `resume`: if we have to recover recommendations from a previous execution.
   - `dataset-related-arguments`: see earlier.
   - (Optional) `-k times`: the number of times each recommendation might be executed 

With these parameters, the different algorithms execute, and the following output is produced:
   - A recommendation file for each recommendation loop execution. The name format for this file is: ` algorithmname-iter.txt`, where algorithm name shows the used algorithm and its parameters. In case the optional parameter `k` is not used, the `iter` value shall be equal to `0`. If `k`is used (and each algorithm is executed several times), it represents the execution number for that algorithm.
   - The algorithm ranking in the comparison, in a file named `algorithms-metric-ranking.txt`, where `algorithms` is the name of the JSON configuration file, and `metric` is each one of the considered metrics: the clickthrough rate in the `stream` case, and the number of iterations / cumulative recall in the rest of cases. The file contains, on each line, an algorithm-value pair, sorted by descending metric value. An example can be observed below:
```
Algorithm	recall
club-erdos-0.01-1.0-ignore	0.05313432835820896
club-erdos-0.01-2.0-ignore	0.03295522388059702
club-erdos-0.01-5.0-ignore	0.03295522388059702
club-erdos-0.01-0.5-ignore	0.02591044776119403
club-erdos-0.1-1.0-ignore	0.008238805970149254
club-erdos-0.1-5.0-ignore	0.0039402985074626865
club-erdos-0.1-2.0-ignore	0.0039402985074626865
```

### Recommendation
Given no warmup, this program executes a set of recommendation algorithms. It is executed as:
```
java -jar IRBandits.jar rec type-of-dataset algorithms input output end-condition resume dataset-related-arguments (-k times -interval interval)
```
where the command line arguments are:
   - `type-of-dataset`: see the earlier type of dataset configuration.
   - `algorithms`: a JSON file containing the possible hyperparameter configurations of algorithms to consider.
   - `output`: the directory in which to store the output.
   - `end-condition`: the end condition for the recommendation. Depending on its value, several possibilities:
       - `end-condition = 0.0`: ends when no user can be recommended anything.
       - `0.0 < end-condition < 1.0`: ends when it has discovered a fraction of the positive ratings equal to `end-condition`.
       - `end-condition > 1.0`: executes it for a number of iterations equal to `end-condition`
   - `resume`: if we have to recover recommendations from a previous execution.
   - `dataset-related-arguments`: see earlier.
   - (Optional) `-k times`: the number of times each recommendation might be executed.
   - (Optional) `-interval interval`: this program produces a summary file for each recommendation. This value establishes the amount of iterations between each recorded point in the summary. By default, it records a register in the summary file each 10,000 iterations.

With these parameters, the different algorithms execute, and the following output is produced:
   - A recommendation file for each recommendation loop execution. The name format for this file is: ` algorithmname-iter.txt`, where algorithm name shows the used algorithm and its parameters. In case the optional parameter `k` is not used, the `iter` value shall be equal to `0`. If `k`is used (and each algorithm is executed several times), it represents the execution number for that algorithm.
   - A summary of the executions of a single algorithm, averaged over the different times the algorithm has been executed. It is named as `algorithm-summary.txt` with `algorithm` being the algorithm name. Each line, the algorithm contains, tab-separated, in the following order, the iteration number and the different metric values. We show next an example of this file:
```
Iteration	recall	gini
100000	0.004083192282072049	0.4030777503476343
200000	0.009147512185132834	0.3103410430621766
300000	0.013760880805087986	0.2676407025492733
400000	0.018585973003067024	0.24228543736198463
500000	0.022876422581124423	0.2244776744496217
```

### Validation with warm-up
This program is similar to the Validation one, but it takes some warm-up data. It is executed as:
```
java -jar IRBandits.jar warmup-valid type-of-dataset algorithms input output end-condition resume training partition-params dataset-related-arguments (-k times - type type)
```
where the command line arguments are:
   - `type-of-dataset`: see the earlier type of dataset configuration.
   - `algorithms`: a JSON file containing the possible hyperparameter configurations of algorithms to consider.
   - `input`: file containing the dataset.
   - `output`: the directory in which to store the output.
   - `end-condition`: the end condition for the recommendation. Depending on its value, several possibilities:
       - `end-condition = 0.0`: ends when no user can be recommended anything.
       - `0.0 < end-condition < 1.0`: ends when it has discovered a fraction of the positive ratings equal to `end-condition`.
       - `end-condition > 1.0`: executes it for a number of iterations equal to `end-condition`
   - `resume`: if we have to recover recommendations from a previous execution.
   - `training`: the route to a recommendation file containing the warm-up data. The format of the file must be the same as the one for recommendation output files.
   - `partition-params`: A series of parameters related to the partitioning of the warm-up data. The partition of this data is always temporal, i.e. it considers the order of appearance of the (user, item) pairs in the `training` file. The first registers are taken as input for the different recommendation algorithms, and some of the rest as test. The parameters are the following:
       - `test-type`: it determines how much ratings in `training` we take as the validation set. If it takes the `fixed` value, every rating in the warm-up which is not used as training data is used as test. If it is equal to `variable` we divide the warm-up data in equal parts. Then, we take the first part, and we split it in training and validation to form the first split. Then, we take the first and the second parts together, and we split them in training and validation to form the second split. The procedure continues until we take the whole data, and we divide it in training and test to form the last split.
       - `numParts`: the number of partitions to consider. In case this value is negative, we consider the set of positively-rated user-item pairs to apply the partition. (for example, when, `test-type` is `variable`, we divide the warm-up data so, each split, we add the same amount of positively-rated user-item pairs). 
       - `percTrain`: it determines the amount of registers on each split that we take as training, and we use the rest as test. In case `numParts` is negative, this percentage is determined so that `percTrain` of the positive ratings in the split are in the training set, and the rest in the validation set.
     
     For example, if `testType = fixed`, `numParts = 5` and `percTrain = 0.1`, then, we shall obtain five partitions, taking 10%,20%,30%,40% and 50% of the warm-up data as training, and the remaining 90%,80%,70%,60% and 50% of the data as the validation set. Otherwise, if `testType = variable`, `numParts = 5` and `percTrain = 0.1`, then, we shall obtain five partitions, containing 20%,40%,60%,80% and 100% of the warm-up data, respectively, and 10% of each partition shall be used as warm-up data, and the rest as the validation set. When `numParts` is negative, the previous amounts are always computed over the set of positive ratings (all negative ratings between ratings go to one set or another depending on where the next positive rating belongs).

   - `dataset-related-arguments`: see earlier.
   - (Optional) `-k times`: the number of times each recommendation might be executed.
   - (Optional) `-type type`: In order to update the algorithms, we can decide whether to use only known data (i.e. data present in the original dataset) or all data.
       -  `onlyratings`: removes all user-item pairs in the warm-up which do not appear in the original dataset.
       -  `full`: uses the warm-up data as it is.   

The output of this program is identical to that of the Validation one, with the exception that a new directory is created for each partition (identified by number).

### Recommendation with warm-up
This program is similar to the Recommendation one, but it takes some warm-up data. It is executed as:
```
java -jar IRBandits.jar warmup-rec type-of-dataset algorithms input output end-condition resume training numParts dataset-related-arguments (-k times -percTrain percTrain -type type -interval interval)
```
where the command line arguments are:
   - `type-of-dataset`: see the earlier type of dataset configuration.
   - `algorithms`: a JSON file containing the possible hyperparameter configurations of algorithms to consider. In this case, the JSON needs to have an array of JSON arrays. Each array contains the configuration of the algorithms that shall be executed for each one of the splits.
   - `input`: file containing the dataset.
   - `output`: the directory in which to store the output.
   - `end-condition`: the end condition for the recommendation. Depending on its value, several possibilities:
       - `end-condition = 0.0`: ends when no user can be recommended anything.
       - `0.0 < end-condition < 1.0`: ends when it has discovered a fraction of the positive ratings equal to `end-condition`.
       - `end-condition > 1.0`: executes it for a number of iterations equal to `end-condition`
   - `resume`: if we have to recover recommendations from a previous execution.
   - `training`: the route to a recommendation file containing the warm-up data. The format of the file must be the same as the one for recommendation output files.
   - `numParts`: the number of partitions to consider. In case this value is negative, we consider the set of positively-rated user-item pairs to apply the partition. 
   - `dataset-related-arguments`: see earlier.
   - (Optional) `-k times`: the number of times each recommendation might be executed.
   - (Optional) `-interval interval`: this program produces a summary file for each recommendation. This value establishes the amount of iterations between each recorded point in the summary. By default, it records a register in the summary file each 10,000 iterations.
   - (Optional) `-type type`: In order to update the algorithms, we can decide whether to use only known data (i.e. data present in the original dataset) or all data.
       -  `onlyratings`: removes all user-item pairs in the warm-up which do not appear in the original dataset.
       -  `full`: uses the warm-up data as it is.    
   - (Optional) `-percTrain percTrain`: By default, the data from the warm-up file is equally divided in `numParts`, and, for each partition `j`, parts `0` to `j` are taken as training. However, if this parameter is present and takes values between 0 and 1, partition `0` shall contain the first `percTrain` user-item pairs, partition `j` shall contain the first `(j+1)*percTrain` fraction of user-item pairs in the warm-up data (with `j` going from `0` to `numParts-1`). If `numParts` is negative, `percTrain` refers to the fraction of positively rated user-item pairs.

The output of this program is identical to that of the Recommendation one, with the exception that a new directory is created for each partition (identified by number).

### Training statistics
This program obtains the statistics for the training data (and the partitions).
It is executed as:
```
java -jar IRBandits.jar train-stats type-of-dataset input training numParts dataset-related-arguments (-percTrain percTrain)
```
where
   - `input`: file containing the dataset.
   - `training`: the route to a recommendation file containing the warm-up data. The format of the file must be the same as the one for recommendation output files.
   - `numParts`: the number of partitions to consider. In case this value is negative, we consider the set of positively-rated user-item pairs to apply the partition. 
   - `dataset-related-arguments`: see earlier.
   - (Optional) `-percTrain percTrain`: By default, the data from the warm-up file is equally divided in `numParts`, and, for each partition `j`, parts `0` to `j` are taken as training. However, if this parameter is present and takes values between 0 and 1, partition `0` shall contain the first `percTrain` user-item pairs, partition `j` shall contain the first `(j+1)*percTrain` fraction of user-item pairs in the warm-up data (with `j` going from `0` to `numParts-1`). If `numParts` is negative, `percTrain` refers to the fraction of positively rated user-item pairs.

In this case, the output is displayed on the standard output. It shows a table containing a) the general information for the dataset (number of users, items, ratings and relevant ratings), and b) for each split, the split number, the number of recommendations, the number of ratings and the number of relevant ratings.

### Algorithm files
In order to execute different configurations, we include in the `config` folder examples of configuration files for the different algorithms. In this case, we use JSON files. We include an example for each algorithm, which can be consulted.

### Random seed
It is possible to set a random seed for the experiments, so that the selection of users and other random choices are the same when the experiment is repeated. For that purpose, in the output directory, just add a file named `rngseedlist` (without any file extension) containing the a list of seeds (must be equal to the optional `k` value in the different programs), and set the parameter `resume` to true.

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
