# TimeSeriesRegularization



# The Big Todo List
## Dataset Stuff 

 - [x] NGAFID MC 2021 IAAI Binary
 - [ ] NGAFIC MC 2021 IAAI Multiclass 
 - [x] TEP 
 - [ ] Baydogan's Archive 
 - [ ] 

## Competitor Methods Stuff 
 - [ ] Resnet 
 - [ ] ROCKET 

# Datasets 

## NGAFID MC 2021 IAAI 
Both as binary class and multiclass 

## TEP 

### Compare with 
TEP results https://www.sciencedirect.com/science/article/pii/S0959152412001503 

## Baydogan's Archieve 

http://www.mustafabaydogan.com/multivariate-time-series-discretization-for-classification.html

## multivariate UEA Time Series Classification datasets
https://www.timeseriesclassification.com/

# Model Methods 
## Competitors 
https://link.springer.com/content/pdf/10.1007/s10618-019-00619-1.pdf
Echo State Networks 
Convolutional Networks 
 - Resnet 
 - Fully Convolutional (no skip connections) 
 - Multi Scale Convolutional Network 
 - TAPNet?? https://github.com/xuczhang/tapnet
 - InceptionTime??
 
 ROCKET??? https://www.sktime.org/en/stable/examples/rocket.html
 
Multi Layer Perceptron (Don't bother) 
Auto Encoders 
Non Deep Methods - Figure out more of these 
 - classic nearest neighbor coupled with DTW and a warping window (WW)
 - COTE/HIVE-COTE 

Read the bakeoff for MTS 
https://link.springer.com/content/pdf/10.1007/s10618-020-00727-3.pdf

## Ours 
ConvMHSA 
 - Basic 
 - Positional Embedding + Tanh 
 - Convolutional Skip Connections 
 - 
