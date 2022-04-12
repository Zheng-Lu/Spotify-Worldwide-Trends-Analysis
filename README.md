# Spotify Worldwide Trends Analysis
Zheng Lu, Judy Zhu

## Table of Contents
- [Overview](#Overview)
- [Introduction](#Introduction)
    * [Context and motivation](#Context-and-motivation)
    * [Objectives](#Objectives)
- [Data](#Data)
    * [Data provenance](#Data-provenance)
    * [Data description](#Data-description)
    * [Data processing](#Data-processing)
- [Exploration and analysis](#Exploration-and-analysis)
- [Discussion and conclusions](#Discussion-and-conclusions)
    * [Summary of findings](#Summary-of-findings)
    * [Evaluation of own work: strengths and limitations](#Evaluation-of-own-work-strengths-and-limitations)
    * [Comparison with any other related work](#Comparison-with-any-other-related-work)
    * [Improvements and extensions](#Improvements-and-extensions)

# Overview
This study is based on the "Worldwide Daily Song Ranking 2017-2018" dataset from Spotify -- a large music-streaming service provider. We first cleaned and wrangled the dataset with more informational columns. We were able to draw general conclusions on the correlations between region, timestamp, and the number of streams by colouring the world map according to the number of streams and plotting the trends of the globally top-ranking songs. For deeper analysis, we performed a PCA unsupervised clustering on the difference in the taste of songs between continents. After a failed try in simple linear regression, we added a random forest regression to predict the positions of music tracks. For further improvements, we carried out logistic regression to predict whether a song would be popular based on additional datasets and audio features. 

# Introduction

## Context and motivation
Due to the widespread use of the Internet, music streaming plays a critical role in our daily lives. In this report, we will analyse a set of data from one of the most extensive audio streaming and media services providers -- Spotify, containing daily rankings of 200-most-listened songs. The study may help the companies locate their customers and song taste based on regions and time. Its results may also give insights to music creators and song recommendation AIs into the different aspects that may affect the popularity of music tracks. 

## Objectives
We are setting out to answer the following questions:
- How does the number of streams distributed around the world?
- How do the positions of the global top-ranking song change over the year?
- Do regions share the same top-ranking artists or songs?
- Can we predict the position of a music track?
- Can we predict whether a music track would become popular based on its audio features?

# Data

# Data provenance
The data used for this project were downloaded from Kaggle's "Spotify's Worldwide Daily Song Ranking (Version 3)" as a CSV file. The dataset was crawled by Eduardo 4 years ago from the website Spotify.Charts **(cite)**. The codes provide a copyleft license, GNU General Public License v3.0 **(cite)**, which includes permissions for commercial use, modification, distribution, patent use and private use. 

For the extension, we used the official Spotify web API to complement the song ranking dataset till the end of 2018 and acquire additional data on audio features. As required on the website for Spotify.Chart developers **(cite)**, we authenticated against their API by “client id” and “secret id” from Spotify Premium account before getting the data. 

## Data description
The main data from Spotify is a daily record of the top 200 playlists global and in 53 regions across the world. It contains 3441197 rows and 7 columns, including the position on charts (*Position*), name of the song (*Track Name*), artist of the song (*Artist*), the number of streams (*Streams*), the website address of the song (*URL*), the date for the record (*$Date*) and the region code (*Region*).

The audio features is also provided by a csv file, with 27802 rows and 26 columns providing 9 features: *duration(ms)*, *acousticness*, *danceability*, *energy*, *instrumentalness*, *liveness*, *tempo*, *valence* and *loudness* for each song.

---
|Attribute | Mean | Std Dev |Description|
|--- | --- | --- | --- |
|1 - Acousticness (float) |0.34|0.34| A confidence measure from 0.0 to 1.0 of whether the track is acoustic.|
|2 - Danceability (float)|0.58|0.18| Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.|
|3 - duration_ms (int)| 212546| 124320| Duration of the track in ms|
|4 - energy (float)|0.57 |0.26 | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.|
|5 - instrumentalness (float) |0.23 |0.36 | Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. 
|6 - key (int)|5.24| 3.60| The estimated overall key of the track.|
|7 - liveness (float)|0.19 | 0.17 |Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.| 
|8 - loudness| -9.94 | 6.50| The overall loudness of a track in decibels (dB)
|9 - mode (int)| 0.61| 0.49| Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived.|
|10 - speechiness (float)| 0.11| 0.12| Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.
|11 - tempo (int)|119.60 |30.15 | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
|12 - time signature (int)| 3.88| 0.51| An estimated overall time signature of a track.| 
|13 - valence (float) | 0.44 | 0.26| A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).| 
---

## Data processing
For the song-ranking dataset, we first dropped the NaN rows, mostly the songs taken down, and the *URL* column since we will not use it in the analysis. Then, we split the *Date* column into *Year*, *Month* and *Weekday* columns, where the *Weekday* is the date of a week calculated by the python package *matplotlib.dates*. An additional column *Continent* is generated according to region codes by the package *geopandas*. We replaced the abbreviations of continents with their full names and distinguished the records with the same track name but different artists by renaming them differently.

For the extension part, we merged the two datasets and receive the record from the start of 2017 to the end of 2018. The dataset of the audio features is generally well structured. We dropped the only 5 NaN rows and transferred *Position* and *Streams* column values into numeric type for further analysis.

[Back to Top](#Table-of-Contents | width=12)

# Exploratory Data Analysis
Our exploration of the data is divided into two parts: the first part analysing the difference in music tastes across the world; the second part focusing on globally top-ranking songs and their patterns. For both parts, we have applied a machine learning technique to look into the problem.

We first visualised the distribution of streams across the world by drawing a pie chart, where the regions with streams in the upper 0.8 quantiles are displayed in the chart while the rest are integrated into the *Other* part. We observed that the United States of America users dominated the market, taking up more than a quarter of the total streams. This disparity may be due to the difference in population and lifestyle. The US has the third-largest population in the world, around 325.14 million in 2017 **(cite)**, and also a wide use of mobile phones and audio devices. The number was followed by the United Kingdom, Brazil, Mexico and Germany, which all provided around 6% to 7% of the total streams. Those countries also have similar advantages. 

![](images/pie-chart.png)

Then, we focused on the global trend of Spotify music. Filtered by the global region, the data provides the number of streams of 200 top-ranking songs for each day. We plotted the time against the position of the 10 songs with the highest number of streams. We observed that all the music tracks had at least a period reaching the top 20 positions. However, since July, almost all popular songs have gradually lost their leading positions. Those relatively short lifespans of the top-ranking songs may be due to the fast pace of current life so that people frequently change their song tastes over time. Also, interestingly, there is a significant decrease in positions for all the 10 songs at Christmas time. During this period, top songs include $"All I Want for Christmas Is You"$ and $"Home for Christmas"$ for lots of people played those songs for celebration. After that, their positions rose back and reached local maxima during New Year. 

![](images/Global-top10-f.png)

Next, we looked into the question: Do people in different regions share the same top-ranking songs? In this section, each song in the dataset was considered to be a feature and was calculated by its streams over a region's total number of streams. A more significant percentage indicated the song's higher popularity in that particular region. We then used the unsupervised principal component analysis (PCA) to determine the most important features and reduce the dimensions. Thus our next step was to determine how many principal components should be included in our graph to be most informative.  

![](images/pca3.png)
![](images/kmeans.png)

By plotting out the number of principal components against the variance of data, we observed an "elbow" at PC3 where the gradient changed sharply. The rule of thumb in our lecture notes would suggest that we use 3 PCs to describe the dataset in our following analysis.

![](images/3d-cluster-f.png)

As a result, the 3-dimensional PCA clustering was plotted by regions according to the three PCs and coloured based on their continents. We could easily see the clusters of data points. For a more rigorous result, we used K-means algorithms to determine the suitable number of clusters. K is plotted against the mean squared error (MSE) by the formula: 
[//]: # (MSE = \frac{1}{\:n}\sum _{k=1}^K\sum _{i\in C_k}\:\parallel x_i-m_k \parallel ^2)
where *m_k* is the centre of the *k*th cluster, *n* is the total number of points, and *|$\dot$|* is the Euclidean norm. From the plot, we can then observe that the elbow is around $K=3$, implying that 3 is a reasonably good number of clusters.

Looking back into the PCA clustering plot, we can see the three clusters: two large ones on the positive and negative PC1 axis and a single point far away from all the others. Hence, we concluded that the Spotify users in Europe, Asia and Oceania tend to share similar music tastes, different from those in the American continent. Also, Luxembourg alone has the unique favour for top songs from the rest of the world. There are also 2 outliers marked in the figure: Spain, despite being a European country, has similar favour in songs with most South American regions, for they have a common speaking language, Spanish, and Latin American culture. Brazil, a country in South America, has its music taste more similar to the cluster of Eurasia than America. 

Finally, to answer whether we can predict music tracks' popularity, we focused on the global data and tried to generate a linear regression model. We used the multi-output regression since we can use the artists and streams to predict the rankings simultaneously. Again, PCA is applied to reduce the dimensions during the process of fitting and standardising the data. However, after plotting the linear regression of the actual versus prediction data, we observed that the outcome did not come out to be proportional to the change in inputs, and we calculated the root mean square error (RMSE) > 30. The non-linearity of data prevented our model from getting good predictions. 

![](images/comparison-regression.png)

|                                                              | **R square score** |             | **RMSE**     |             | **MAE**      |             |
|:-------------------------------------------------------------|:-------------------|:------------|:-------------|:------------|:-------------|:------------|
| **Regression model**                                         | **Training**       | **Testing** | **Training** | **Testing** | **Training** | **Testing** |
| **Multivariate Linear Regression with Scaling and PCA only** | 0.7165             | 0.7071      | 29.58        | 30.163      | 23.052       | 23.493      |
| **Multivariate Linear Regression with random forest**        | 0.9931             | 0.9479      | 4.581        | 12.654      | 3.04         | 8.37        |

Therefore, we introduced another regression method, the random forest -- a supervised machine learning algorithm that is an ensemble of decision trees. Firstly, the samples used to generate the decision trees are chosen by the bootstrap method, where we took various random sub-samples from the dataset with replacement. Then, we constructed a decision tree for each sample and got a prediction result. Here the mean squared error (MSE) is calculated by the formula: 
[//]: # (MSE = \frac{1}{n}\sum _{i=1}^n\:\left(y_i-\hat{y_i}\right)^2)
where *n* is the size of sample, *y_i* is the actual value for data point *i* and *\hat{y_i}* is the predicted value by the model, is used to decide which branch to choose from each node. Especially, in order to make each tree unique enough to have own features, the decision trees are not pruned in the training. After a large amount of calculations, we received the prediction value from all the decision trees and averaged for the final result. The actual versus prediction line is drawn.

For a more precise prediction, we divided the model into three parts: "average rank position", "highest rank position", and "lowest rank position". After tuning and scaling the data by PCA features, we applied the above process of the multi-output and random forest regressor, and immediately received a model with testing scores every close to 1 and RMSE = 12.654. The visualisation of the actual versus prediction data also shows a reasonably good fit.

![](images/random-forest-f.png)
