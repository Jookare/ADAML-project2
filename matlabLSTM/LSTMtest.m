%% ADAML Project 2 - LSTM model for daily average humidity in Delhi
% Lasse Johansson (following a tutorial for numerical Matlab LSTM)
clc; clear all; close all; 

%% Load the data and make simple visuals
%load data
load('data\delhiClim.mat') %training set
load('data\delhiClimTest.mat')% testing set

data(1:4)
%View the number of channels. To train the LSTM neural network, each sequence must have the same number of channels.
numChannels = size(data{1},1)
%Visualize the first few sequences in a plot.
figure
tiledlayout(2,2)
for i = 1:4
    nexttile
    stackedplot(data{i}')
    xlabel("Time Step")
end

numObservations = numel(data)
dataTrain = data;%(idxTrain);
dataTest = test_data;%(idxTest);

%% Prepare Data for Training
%To forecast the values of future time steps of a sequence, specify the targets as the training sequences with values shifted by one time step. In other words, at each time step of the input sequence, the LSTM neural network learns to predict the value of the next time step. The predictors are the training sequences without the final time step.
for n = 1:numel(dataTrain)
    X = dataTrain{n};
    XTrain{n} = X(:,1:end-1);
    TTrain{n} = X(:,2:end);
end

%normalize the data. testing data normalization based on the training set.
muX = mean(cat(2,XTrain{:}),2);
sigmaX = std(cat(2,XTrain{:}),0,2);

for n = 1:numel(XTrain)
    XTrain{n} = (XTrain{n} - muX) ./ sigmaX;
    TTrain{n} = (TTrain{n} - muX) ./ sigmaX;
end

for n = 1:size(dataTest,1)
    X = dataTest{n};
    XTest{n} = (X(:,1:end-1) - muX) ./ sigmaX;
    TTest{n} = (X(:,2:end) - muX) ./ sigmaX;
end


%% Define LSTM Neural Network Architecture
layers = [
    sequenceInputLayer(numChannels)
    lstmLayer(128)
    fullyConnectedLayer(numChannels)
    regressionLayer];

% Specify Training Options
options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=0);

%% Train
net = trainNetwork(XTrain,TTrain,layers,options);
figure;
plot(net)

%% forecast - Opoen loop
idx = 1;
X = XTest{idx};
T = TTest{idx};
%Initialize the RNN state by first resetting the state using the resetState function, then make an initial prediction using the first few time steps of the input data. Update the RNN state using the first 75 time steps of the input data.
net = resetState(net);
offset = 2;
[net,~] = predictAndUpdateState(net,X(:,1:offset));
%To forecast further predictions, loop over time steps and update the RNN state using the predictAndUpdateState function. Forecast values for the remaining time steps of the test observation by looping over the time steps of the input data and using them as input to the RNN. The first prediction is the value corresponding to the time step offset + 1.
numTimeSteps = size(X,2);
numPredictionTimeSteps = numTimeSteps - offset;
Y = zeros(numChannels,numPredictionTimeSteps);

for t = 1:numPredictionTimeSteps
    Xt = X(:,offset+t);
    [net,Y(:,t)] = predictAndUpdateState(net,Xt);
end

%convert back to original scale


%Compare the predictions with the target values.
figure
t = tiledlayout(numChannels,1);
title(t,"Open Loop Forecasting")

Tb = T.*sigmaX + muX;
Yb = Y.*sigmaX + muX;
rmse = comp_RMSE(Tb,Yb,offset)
for i = 1:numChannels
    nexttile
    plot(Tb(i,:))
    hold on
    plot(offset:numTimeSteps,[Tb(i,offset) Yb(i,:)])
    ylabel("Channel " + i)
end

xlabel("Time Step")
nexttile(1)
legend(["Input" "Forecasted"])

%% closed loop forecast
%Closed loop forecasting predicts subsequent time steps in a sequence by using the previous predictions as input. In this case, the model does not require the true values to make the prediction. For example, say you want to predict the value for time steps  through  of the sequence using data collected in time steps 1 through  only. To make predictions for time step , use the predicted value for time step  as input. Use closed loop forecasting to forecast multiple subsequent time steps or when you do not have true values to provide to the RNN before making the next prediction.
%Initialize the RNN state by first resetting the state using the resetState function, then make an initial prediction Z using the first few time steps of the input data. Update the RNN state using all time steps of the input data.
net = resetState(net);
[net,Z] = predictAndUpdateState(net,X(:,1:offset));
%To forecast further predictions, loop over time steps and update the RNN state using the predictAndUpdateState function. Forecast the next 200 time steps by iteratively passing the previous predicted value to the RNN. Because the RNN does not require the input data to make any further predictions, you can specify any number of time steps to forecast.
Xt = X(:,offset);
Y = zeros(numChannels,numPredictionTimeSteps);

for t = 1:numPredictionTimeSteps
    [net,Y(:,t)] = predictAndUpdateState(net,Xt);
    Xt = Y(:,t);
end
%Visualize the forecasted values in a plot.
%numTimeSteps = offset + numPredictionTimeSteps;

figure
t = tiledlayout(numChannels,1);
title(t,"Closed Loop Forecasting")


Yb = Y.*sigmaX + muX;
rmse = comp_RMSE(Tb,Yb,offset)
for i = 1:numChannels
    nexttile
    plot(Tb(i,:))
    hold on
    plot(offset:numTimeSteps,[Tb(i,offset) Yb(i,:)])
    ylabel("Channel " + i)
end

xlabel("Time Step")
nexttile(1)
legend(["Input" "Forecasted"])

%% utility function to compute RMSE after the offset
function val = comp_RMSE(Tb, Yb, offset)
  L = length(Tb);
  N = L-offset;
  sum =0;
  for i =offset:L
     e = Tb(i)-Yb(i);
     sum =sum+ e*e/N; 
  end    
 
 val = sqrt(sum);
end
