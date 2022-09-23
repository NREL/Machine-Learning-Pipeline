%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC

function obj = featureSelection(obj,FeatureEngineeredX, y, varargin)
%FEATURESELECTION performs feature selection on FeatureEngineeredX and returns the reduced X
%   [FeatureSelectedX, l, Fit, coef] =  FEATURESELECTION(FeatureEngineeredX,y)
%   The user can specify which feature selection algorithm they would like
%   to use, otherwise it will default to returning the Feature Engineered X. The algorithm
%   will then identify the most important feautres and the returned
%   FeatureSelectedX will be a reduced feature set of FeatureEngineeredX. This is the step
%   after the FeatureEngineering step in the pipeline algorithm.
%   Inputs:
%       FeatureEngineeredX: Table or array of features
%       y: Array of the dependent variable
%   Ouputs:
%       FeatureSelectedX (table): A reduced feature set of predictors tha
%       should be used to predict y in the final step of the algorithm.

%Get Feature Selection Algorithm
maskFSAlgorithm = strcmp(varargin, 'FeatureSelectionAlgorithm');
isFeatureSelectionSpecified = any(maskFSAlgorithm);

%Perform Feature Selection if algorithm specified
if isFeatureSelectionSpecified
     maskFSAlgorithmValue = logical([0 maskFSAlgorithm]);
     algorithm = varargin{maskFSAlgorithmValue};
     obj.FS_Algorithm = algorithm;
     %Use LASSO alogrithm a form of penalized regression
     if contains(algorithm, 'LASSO')
        [l, Fit] = lasso(FeatureEngineeredX, y, 'CV', 10);
        coef = l(:, Fit.Index1SE);
        FeatureSelectedX = FeatureEngineeredX(:,coef'>0.001 | coef'<-0.001);
        obj.FS_Coefs = coef;
     %Use a ranking based on F-test scores
     elseif contains(algorithm, 'fsrftest')
         idx = fsrftest(FeatureEngineeredX,y);
         FeatureSelectedX = FeatureEngineeredX(:,idx'<6);
         obj.FS_Coefs = idx;
     %Feature Selection based on neighborhood component analysis
     elseif contains(algorithm, 'fsrnca')
         mdl = fsrnca(FeatureEngineeredX, y);
         FeatureSelectedX = FeatureEngineeredX(:,mdl.FeatureWeights'>1);
         obj.FS_Coefs = mdl.FeatureWeights;
     %Use a ranking based on LaPlacian scores
     elseif contains(algorithm, 'fsulaplacian')
         idx = fsulaplacian(FeatureEngineeredX);
         FeatureSelectedX = FeatureEngineeredX(:, idx'<6);
         obj.FS_Coefs = idx;
     %Use a form of penalized KNN algorithm for Feature Selection
     elseif contains(algorithm, 'relieff')
         [idx, ~] = relieff(FeatureEngineeredX, y, 5);
         FeatureSelectedX = FeatureEngineeredX(:,idx'<6);
         obj.FS_Coefs = idx;
     end
else
%     [l, Fit] = lasso(FeatureEngineeredX, y, 'CV', 10);
%     coef = l(:, Fit.Index1SE);
%     FeatureSelectedX = FeatureEngineeredX(:,coef'>0.001 | coef'<-0.001);
%     obj.FS_Coefs = coef;
    FeatureSelectedX = FeatureEngineeredX;
end
obj.FeatureSelectedX = FeatureSelectedX;
end