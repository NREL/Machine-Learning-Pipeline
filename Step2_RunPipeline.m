%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC
%%% Andrew Schiek 2022

close all; clear all;

%% Train Pipeline with MIT Data
addpath('Data')
load MIT_Variables.mat
X_MIT = table2array(X_MIT);
y_MIT = table2array(y_MIT);

%%% Split Train and Test Sets on SOH Cutoff
% X_Train = X_MIT(y_MIT<0.96);
% X_Test = X_MIT(y_MIT>=0.96);
% y_Train = y_MIT(y_MIT<0.96);
% y_Test = y_MIT(y_MIT>=0.96);
%%%

%Split Train and Test Sets randomly
testSplit = 0.2; %Percentage of Data Set used for test. Change this accordingly.
rngSeed = 122599;
rng(rngSeed);
cv = cvpartition(height(X_MIT), 'Holdout', testSplit);
idxTrain = training(cv);
X_Train = X_MIT(idxTrain,:);
y_Train = y_MIT(idxTrain,:);
idxTest = test(cv);
X_Test = X_MIT(idxTest,:);
y_Test = y_MIT(idxTest,:);

%Create Pipeline Class called pipe
pipe = Pipeline();
%Run Feature Engineering
pipe = pipe.featureEngineering(X_Train, y_Train, 'Normalization', 'Z', 'FeatureEngineeringAlgorithm', 'PCA');
%Run Feature Selection
pipe = pipe.featureSelection(pipe.FeatureEngineeredX, y_Train, 'FeatureSelectionAlgorithm', 'fsrnca');
%Train Model
[pipe, FitResult] = pipe.fit(pipe.FeatureSelectedX, y_Train, 'FitAlgorithm', 'RandomForest');
%Test Model on Test Set
TestResult = pipe.evaluate(X_Test, y_Test, 'Normalization', 'Z');

