%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC
%%% Andrew Schiek 2022

function [obj, FitResult] = fit(obj, FeatureSelectedX, y, varargin)
%FIT fits X to y and returns the predicted y values and the FitResult
%   [obj, FitResult] = FIT(FeatureSelectedX, y)
%   The user can specify which algorithm they would like
%   to use to fit the data, otherwise it will default to the GPR algorithm. The algorithm
%   will then fit the data in FeatureSelectedX to y. The ypred will be the
%   predicted values of y and the model will be produced in FitResult.
%   This is the step after Feature Selection in the pipeline algorithm.
%   Inputs:
%       FeatureSelectedX (table): Table of features
%       y (array): Array of the dependent variable
%   Ouputs:
%       ypred (array): predicted values for the dependent variable y.
%       FitResult: Instance of the FitResult class.

    %Get Fit Algorithm
    maskFitAlgorithm = strcmp(varargin, 'FitAlgorithm');
    isFitAlgorithmSpecified = any(maskFitAlgorithm);
    
    %Fit FeatureSelectedX data to predict y
    if isFitAlgorithmSpecified
        maskFitAlgorithmValue = logical([0 maskFitAlgorithm]);
        algorithm = varargin{maskFitAlgorithmValue};
        obj.Fit_Algorithm = algorithm;
        
        %Fit using GPR algorithm
        if contains(algorithm, 'GPR')
            obj.Model = fitrgp(FeatureSelectedX,y, 'KernelFunction', 'rationalquadratic');
            obj.ypred = resubPredict(obj.Model);
        %Fit using Random Forest Algorithm
        elseif contains(algorithm, 'RandomForest')
            obj.Model = fitrtree(FeatureSelectedX, y, 'OptimizeHyperparameters', 'all');%, 'CrossVal', 'on');
            obj.ypred = obj.Model.predict(FeatureSelectedX);
        end
        
    %Default to GPR if no algorithm is specified or an unusable algorithm is specified
    else
        obj.Fit_Algorithm = 'GPR';
        obj.Model = fitrgp(FeatureSelectedX,y, 'KernelFunction', 'rationalquadratic');
        obj.ypred = resubPredict(obj.Model);
    end
    
    %Get training set error
    FitResult.MSE = 1/length(y) * sum((y-obj.ypred).^2);
    FitResult.RMSE = sqrt(FitResult.MSE);
    FitResult.MAPE = 1/length(y) * sum(abs((y-obj.ypred)./y));
end