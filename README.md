# :robot: BitcoinPrediction :car:

## :arrow_forward: Introduction
Bitcoin represents a modern way to transport money from peer to peer. Its value may change over time based on multiple factors. This project comes with help for predicting the future price.

## :memo: Description

To predict the future open price, it's necessary to know all the dependent variables that may change the outcome. Machine Learning Regression is a technique for investigating the relationship between independent variables or features and a dependent variable. Neural networks are often used for precise predictions. LSTM is used for long short-term memory networks in Deep Learning. It is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies, especially in sequence prediction problems. LSTM has feedback connections, i.e., it is capable of processing the entire sequence of data.
## :computer: Implementation

The dataset consists of multiple rows containing information for past prices. It is normalized by using MinMaxScaler. The neural network is imported from keras.models and its parameters are chosen based on previous outcomes. There are 5 LSTM layers, along with 4 Dropout layers. Data is split into two categories, train and test. Training data is transmitted to the model by using the fit() method. Outcomes represent the predictions made by the model over the test group. The final drawing represents the visual difference between predicted outcomes and real ones.

## :exclamation: Instructions
The application can be started using the build and run command inside IDE.
  
 ## :camera: Graph
<p align="center">
 <img src="https://github.com/Marius2504/BitcoinPrediction/blob/master/bitcoin_generated.png" width="600">
</p>

