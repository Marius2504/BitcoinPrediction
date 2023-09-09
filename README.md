# :robot: BitcoinPrediction :car:

## :arrow_forward: Introduction
Bitcoin represent a modern way to tranport money from peer to peer. Its value may change over time based on multiple factors. This porject comes with help for predicting the future price.  

## :memo: Description

In order to predict the future open price, it's necessary to know all the dependent variables that may change the outcome. Machine Learning Regression is a technique for investigating the relationship between independent variables or features and a dependent variable. Neural networks are often used for precised predictions. LSTM are used for long short-term memory networks in Deep Learning. It is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies, especially in sequence prediction problems. LSTM has feedback connections, i.e., it is capable of processing the entire sequence of data.
## :computer: Implementation

Dataset consists of multiple rows containg information for past prices. It is normalised by using MinMaxScaler. Neural network is imported from keras.models and its parameters are chosen based on prevous outcomes. There are 5 LSTM layers, along with 4 Dropout layers.Data is split in two categories, train and test. Training data is transmited to model by using fit() method. Outcomes represent the predictions made by the model over the test group. The final drawing represent the visual difference between predicted outcomes and real ones.

## :exclamation: Instructions
The application can be started using build and run command inside IDE.
  
 ## :camera: Graph
<p align="center">
 <img src="https://github.com/Marius2504/BitcoinPrediction/blob/master/bitcoin_generated.png" width="600">
</p>

