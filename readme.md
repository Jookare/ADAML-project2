# Humidity Forecasting from Multivariate Climate Data
## Overview
This project focuses on forecasting climate parameters, specifically predicting daily humidity levels using multivariate time series data. The work applies multiple machine learning models for time series forecasting and compares them against a sufficient baseline model. [Read full report](./Report.pdf).

## Dataset 
The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data). It describes daily average meteorological conditions in Delhi, India, from 1st of January 2013 to 24th of April 2017. The dataset contains four parameters: mean temperature [°C],  absolute humidity [g/m³], wind speed [km/h], and mean pressure [atm].

## Models implemented
1. Baseline (Naive) model: Simple approach using historical values as predictions.
2. Recurrent Neural Network (RNN): Captures sequential patterns in the data. [View PDF](./images/networks/RNN.pdf)
3. Long Short-Term Memory (LSTM): Enhances RNN with memory cells to learn long-term dependencies. [View PDF](./images/networks/LSTM.pdf)
4. Full Transformer Model: Employs attention mechanisms for sequence-to-sequence prediction. [View PDF](./images/networks/Transformer.pdf)
5. Decoder-Only Model: Simplifies the Transformer architecture to focus on prediction tasks. [View PDF](./images/networks/Decoder.pdf)

## Key Results
### Table: RMSE for Baseline, RNN, LSTM, TNN, and Decoder Models (best result underlined)
| Horizon | **Baseline** | **RNN** | **LSTM** |   **TNN**   |  **Decoder** |
|:-------:|:------------:|:-------:|:--------:|:-----------:|:------------:|
| 1       | 7.42         | 7.10    | 7.78     | <u>6.90</u> | 7.04         |
| 2       | 9.13         | 8.90    | 8.52     | 7.49        | <u>7.28</u>  |
| 5       | 10.42        | 12.52   | 9.95     | 9.96        | <u>9.39</u>  |
| 10      | 11.26        | 13.93   | 12.09    | 10.64       | <u>10.01</u> |

## Conclusion
The project demonstrates that the accuracy is only slightly better with the deep learning models than with the simple baseline model, showing that humidity (and weather more generally) cannot be predicted accurately several days ahead. 