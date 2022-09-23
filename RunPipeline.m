%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC

close all; clear all;
addpath('Data')

%Create Pipeline Class called pipe
pipe = Pipeline();
%Load Dataset and split with a 20% test set
pipe = pipe.loadData('MIT_Variables.mat', 'TestTrainSplit', 'Percentage', 'TestPercent', 0.2);
%Run Feature Engineering
pipe = pipe.featureEngineering(pipe.X_Train, 'Normalization', 'Z', 'FeatureEngineeringAlgorithm', 'PCA');
%Run Feature Selection
pipe = pipe.featureSelection(pipe.FeatureEngineeredX, pipe.y_Train, 'FeatureSelectionAlgorithm', 'fsrnca');
%Train Model
[pipe, FitResult] = pipe.fit(pipe.FeatureSelectedX, pipe.y_Train, 'FitAlgorithm', 'RandomForest');
%Test Model on Test Set
TestResult = pipe.evaluate(pipe.X_Test, pipe.y_Test, 'Normalization', 'Z');

