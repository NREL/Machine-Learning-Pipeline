%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC
%%% Andrew Schiek 2022

function obj = featureEngineering(obj, X, varargin)
%FEATUREENGINEERING performs feature engineering on X and returns a larger
%FeatureEngineeredX.
%   [FeatureEngineeredX, coef, explained] = FEATUREENGINEERING(X)
%   The user can specify which feature engineering algorithm they would
%   like to use, otherwise it will default to no feature engineering
%   algorithm. The user can also specify if they would like to normalize
%   the data or not and if they want it to be a Z-score or Min-Max
%   normalization. Using the feature engineering algorithm selected this
%   function will return an Engineered Features set that consists of the
%   new features on the end of the original features. This is the second
%   step in the Pipeline algorithm, and loadData should be run first.
%   Inputs:
%       X: Array of features
%   Outputs:
%       FeatureEngineeredX: An array of additional engineered features

    %Get Normalization Method
    maskNorm = strcmp(varargin, 'Normalization');
    isNormSpecified = any(maskNorm);
    
    %Normalize Data if Norm Method Specified
    if isNormSpecified
        maskNormValue = logical([0 maskNorm]);
        NormMethod = varargin{maskNormValue};
        obj.TransformMethod = NormMethod;
        
        %Z-Score Normalize Data
        if contains(NormMethod, 'Z')
            m = [];
            s = [];
            for i=1:width(X)
                m = [m; mean(X(:,i))];
                s = [s;std(X(:,i))];
                X(:,i) = (X(:,i)-mean(X(:,i)))/std(X(:,i));
            end
            obj.TransformMean = m;
            obj.TransformSd = s;
        %Min-Max Normalize Data    
        elseif contains(NormMethod, 'MinMax')
            X = normalize(X, 'range');
        else
            disp('Normalization Method Not Recognized')
        end
    end
    
    %Get Feature Engineering Method
    maskFEAlgorithm = strcmp(varargin, 'FeatureEngineeringAlgorithm');
    isFEAlgorithmSpecified = any(maskFEAlgorithm);
    
    %Perform Feature Engineering if method specified
    if isFEAlgorithmSpecified
        maskFEAlgorithmValue = logical([0 maskFEAlgorithm]);
        algorithm = varargin{maskFEAlgorithmValue};
        obj.FE_Algorithm = algorithm;
        
        %Use PCA for Feature Engineering
        if contains(algorithm, 'PCA')
            [coef,~,~,~,explained,~] = pca(X);
            idx = find(cumsum(explained)>90,1);
            NewFeatures = X*coef(:,1:idx);
            obj.FeatureEngineeredX = [X, NewFeatures];
            obj.FE_Coefs = coef(:,1:idx);
        else
            disp('Feature Engineering Algorithm Not Recognized')
            obj.FeatureEngineeredX = X;
        end
    else
        obj.FeatureEngineeredX = X;
    end
end