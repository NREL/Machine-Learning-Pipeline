%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC
%%% Andrew Schiek 2022

function FitResult = evaluate(obj, X, y, varargin)
%EVALUATE tests the model or refits a model to a test and returns FitResult
%with the testing errors and potential new model.
%   [FitResult] = EVALUATE(X,y)
%   The user can specify if the test data should be normalized or refit. If
%   the test data should be normalized the user must specify either Z
%   (for Z-score) or MinMax. If the test data should be refit then the user
%   must specify either GPR or RandomForest for the algorithm that will be
%   used to refit the test data. The X feature set will then be normalized
%   if specified followed by recieving the same feautre engineering and 
%   feature selection as the training set. If Refit is specified then
%   evaluate will separate the test data into training and test sets, refit
%   a model to the training set, and evaluate that fit on the test set. If
%   refit is not specified then evaluate will perform error analysis on the
%   full X provided. This is the last setp in the Pipeline algorithm.
%   FitResult contains the X and y tested on (after feature engineering and
%   feature selection), X and y splits and the new model if refit is
%   specified, and the MSE, RMSE, and MAPE error metrics.
%   The user can specify which feature engineering algorithm they would
%   like to use, otherwise it will default to no feature engineering
%   algorithm. The user can also specify if they would like to normalize
%   the data or not and if they want it to be a Z-score or Min-Max
%   normalization. Using the feature engineering algorithm selected this
%   function will return an Engineered Features set that consists of the
%   new features on the end of the original features. This is the first
%   step in the Pipeline algorithm.
%   Inputs:
%       X: Table or array of features for testing
%       y: Arrary of dependent variable for testing
%   Outputs:
%       FitResult: A structure containing test data, model, and error

    %Check if there is Normalization required
    maskNorm = strcmp(varargin, 'Normalization');
    isNormSpecified = any(maskNorm);
    
    %Check if we need to refit
    maskRefit = strcmp(varargin, 'Refit');
    isRefitSpecified = any(maskRefit);
    
    %Normalize data if specified
    if isNormSpecified
        maskNormValue = logical([0 maskNorm]);
        NormMethod = varargin{maskNormValue};
        FitResult.TransformMethod = NormMethod;
        
        %Z-Score Normalize
        if contains(NormMethod, 'Z')
            %If refitting normalize with built in method, if not normalize
            %to training mean and standard deviation
            if isRefitSpecified
                X = normalize(X);
            else
                for i=1:width(X)
                    X(:,i) = (X(:,i)-obj.TransformMean(i))/obj.TransformSd(i);
                end
            end
        %Min-Max Normalize data
        elseif contains(NormMethod, 'MinMax')
            X = normalize(X, 'range');
        else
            disp('Normalization Method Not Recognized')
        end
    end
    
    %Replace NaN with zero
    idx = find(all(isnan(X),3));
    if isempty(idx) == 0
        X(idx) = 0;
    end
    
    %Apply Feature Engineering coefficients to test data
    if obj.FE_Coefs
        NewFeatures = X*obj.FE_Coefs;
        X = [X, NewFeatures];
    end
    
    %Apply Feature Selection coefficients/ranks to test data
    if isempty(obj.FS_Coefs) == 0
        if contains(obj.FS_Algorithm, 'LASSO')
            X = X(:,obj.FS_Coefs'>0.001 | obj.FS_Coefs'<-0.001);
        elseif contains(obj.FS_Algorithm, 'fsrnca')
            X = X(:, obj.FS_Coefs'>1);
        else
            X = X(:,obj.FS_Coefs'<6);
        end
    end
    
    %Refit the data if specified
    if isRefitSpecified
        %Get Refit Algorithm
        RefitMethod = obj.Fit_Algorithm;
        
        %Split the test data into test and training sets
        testSplit = 0.4;
        rngSeed = 122599;
        rng(rngSeed);
        cv = cvpartition(length(X), 'Holdout', testSplit);
        idxTrain = training(cv);
        X_train = X(idxTrain,:);
        y_train = y(idxTrain,:);
        idxTest = test(cv);
        X_test = X(idxTest,:);
        y_test = y(idxTest,:);

%         X_train = X(y>0.975);
%         X_test = X(y<=0.975);
%         y_train = y(y>0.975);
%         y_test = y(y<=0.975);
        
        %Refit with GPR
        if contains(RefitMethod, 'GPR')
            FitResult.Model = fitrgp(X_train,y_train, 'KernelFunction', 'rationalquadratic');
            ypred_train = predict(FitResult.Model, X_train);
        %Refit with Random Fores
        elseif contains(RefitMethod, 'RandomForest')
            FitResult.Model = fitrtree(X_train,y_train, 'OptimizeHyperparameters', 'all');
            ypred_train = FitResult.Model.predict(X_train);
        end
        
        %Calculate Training Errors for the Refit
        FitResult.Train_MSE = 1/length(y_train) * sum((y_train-ypred_train).^2);
        FitResult.Train_RMSE = sqrt(FitResult.Train_MSE);
        FitResult.Train_MAPE = 1/length(y_train) * sum(abs((y_train-ypred_train)./y_train));
        ypred = predict(FitResult.Model, X_test);
        FitResult.X_train = X_train; FitResult.y_train = y_train; FitResult.ypred_train = ypred_train;
        FitResult.X_test = X_test; FitResult.y_test = y_test;
        y = y_test;
    else 
        %Use the already fitted model to get y predicted values
        ypred = predict(obj.Model, X);
    end
    
    %Plot ypred vs actual value to evaluate fit
    line = linspace(min([ypred; y])-0.05, max([ypred; y])+0.05);
    figure
    scatter(y, ypred, 'LineWidth', 1.5)
    hold on
    plot(line, line, '-r', 'LineWidth', 1.5)
    xlim([min([ypred; y])-0.05, max([ypred; y+0.05])])
    ylim([min([ypred; y])-0.05, max([ypred; y+0.05])])
    box on
    xlabel('Actual SOH', 'FontSize', 16, 'FontWeight', 'bold')
    ylabel('Predicted SOH', 'FontSize', 16, 'FontWeight', 'bold')
    
    %Calculate error on Test Set
    FitResult.MSE = 1/length(y) * sum((y-ypred).^2);
    FitResult.RMSE = sqrt(FitResult.MSE);
    FitResult.MAPE = 1/length(y) * sum(abs((y-ypred)./y));
    FitResult.ypred = ypred;
    FitResult.TestX = X;

end