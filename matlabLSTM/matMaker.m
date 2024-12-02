clc; clear all; close all; 
% Parameters for reading the CSV file
startColumn = 5; % Specify the starting column index (1-based indexing)
numColumns = 6;  % Number of columns to read

% Read the CSV file, skipping the first line and specified columns
%csvData = csvread('data/DailyDelhiClimateTrain_comma.csv', 1, startColumn - 1); % Adjust for zero-based indexing
csvData = csvread('data/DailyDelhiClimateTest_comma.csv', 1, startColumn - 1);
% Extract the required number of columns
csvData = csvData(:, 1:numColumns); % Select only the desired 6 columns

% Validate the dimensions of the data
[numRows, numCols] = size(csvData);
if numCols ~= numColumns
    error('The specified range of columns does not match the expected number of columns (%d).', numColumns);
end

% Number of partitions
numPartitions = 1;

% Calculate partition sizes
partitionSizes = round(linspace(0, numRows, numPartitions + 1));

% Initialize cell array to hold the partitions
test_data = cell(numPartitions, 1);

% Split the data into approximately equal partitions
for i = 1:numPartitions
    startIdx = partitionSizes(i) + 1; % Start index for the current partition
    endIdx = partitionSizes(i + 1);   % End index for the current partition
    test_data{i} = csvData(startIdx:endIdx, :)'; % Transpose to make columns 6xN
end

% Save the data to a .mat file
save('data\delhiClimTest.mat', 'test_data');
load('data\delhiClimTest.mat')
