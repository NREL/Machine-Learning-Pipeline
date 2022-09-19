%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC
%%% Andrew Schiek 2022

classdef Pipeline
    %Pipline Automates the process of predicting state of health through a ML pipeline.
    %   The final model will predict y from a transformed data set of x.
    %   The x data set will be fed through a feature engineering algorithm and normalized
    %   in the EngineerFeatures function. Then the engineered features will
    %   be fed through a feature selection algorithm in SelectFeatures
    %   function. Finally the fit function will fit a model to the y data
    %   and the model will be stored in Model.
    
    properties %In future/if time look at removing data from pipeline class properties
        X           % X data table
        X_Train     % X training set
        X_Test      % X test set
        FeatureEngineeredX %Largest x data table
        TransformMethod %Method for normalization
        TransformMean %Mean of Z-score normalization
        TransformSd %Standard deviation of Z-score normalization
        FeatureSelectedX %Reduced x data table
        y           % y data table
        y_Train     % y training set
        y_Test      % y test set
        ypred       % y predicted array
        FE_Coefs    % Coefficients for the Feature Engineering Process
        FS_Coefs    % Coefficients for the Feature Selection Process
        FE_Algorithm %Feature Engineering Algorithm
        FS_Algorithm %Feature Selection Algorithm
        Fit_Algorithm %Fit Algorithm
        Model       % RegressionGP class instance
    end
    
    methods
        function obj = Pipeline()
            % Instantiates a DeltaGpr class
            obj.X = []; 
            obj.y = [];
            obj.ypred = [];
            obj.FeatureEngineeredX = [];
            obj.TransformMethod = [];
            obj.TransformMean = [];
            obj.TransformSd = [];
            obj.FeatureSelectedX = [];
            obj.FE_Coefs = [];
            obj.FS_Coefs = [];
            obj.FE_Algorithm = [];
            obj.FS_Algorithm = [];
            obj.Fit_Algorithm = [];
            obj.Model = [];
        end
        %Load and Split Data:
        obj = loadData(obj, Filename, varargin)
        %Perform Feature Engineering:
        obj = featureEngineering(X, varargin);
        %Perform Feature Selection:
        obj = featureSelection(FeatureEngineeredX, y, varargin);
        % Create and optimize a model:
        [obj, FitResult] = fit(obj, FeatureSelectedX, y, varargin)
        % Evaluate a previously optimized model on new data:
        PipelineFitResult = evaluate(obj, y, varargin)
    end
    methods(Static)

    end
end

