%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC
%%% Andrew Schiek 2022
%%% Data being loaded has been processed from the publicly available dataset
%%% in "Data-driven prediction of battery cycle life before capacity 
%%% degradation" by Severson et al. 

function obj = loadData(obj, Filename, varargin)
%LOADDATA loads in a data set stored in Filename and splits the data into
%test and training sets (X_Train, X_Test, y_Train, y_Test)
%   [X, y] = LOADDATA(Filename)
%   The user may specify how they would like to split the data set into
%   test and training sets. Currently supporting options for 'Percentage'
%   and 'SOH' called by 'TestTrainSplit'. If 'Percentage' is selected then
%   the user may input the desired 'TestPercent' (0-1) or it will default
%   to a 20% test set. If 'SOH' is selected the user must put in their
%   desired SOH cutoff for the training set meaning all values below that
%   SOH will be used for testing. In this case if the user does not input a
%   cutoff no splitting will be done. If no 'TestTrainSplit' is selected it
%   will be assumed that the entire data set is the training set.
%   Inputs:
%       Filename: Name of file with predictors saved as a table X, and
%       dependent variable saved as table or array y.
%   Outputs:
%       X: An array of the entire data set of predictors.
%       y: An array of all the dependent variables.

    %Load Filename
    load(Filename)
    X = table2array(X);
    y = table2array(y);
    %Set the original Data X and y
    obj.X = X;
    obj.y = y;

    %Get Normalization Method
    maskSplit = strcmp(varargin, 'TestTrainSplit');
    isSplitSpecified = any(maskSplit);

    %Apply test/train split if specified
    if isSplitSpecified
        %Get split method 
        maskSplitValue = logical([0 maskSplit]);
        SplitMethod = varargin{maskSplitValue};
        
        %Split data based on specified method
        if contains(SplitMethod, 'Percentage')
            %Use a train/test split and check to see if the user specified 
            %a test percent 
            maskTestPercent = strcmp(varargin, 'TestPercent');
            isTestPercentSpecified = any(maskTestPercent);
            
            %Apply the user specified test percent or default it to 80/20
            %Train/Test split
            if isTestPercentSpecified
                maskTestPercentValue = logical([0 maskTestPercent]);
                testSplit = varargin{maskTestPercentValue};
            else
                testSplit = 0.2;
                disp("No 'TestPercent' specified, so using a 20% Test Set")
            end
            
            %Split the data into test and training sets based on test
            %percent
            rngSeed = 122599;
            rng(rngSeed);
            cv = cvpartition(height(X), 'Holdout', testSplit);
            idxTrain = training(cv);
            obj.X_Train = X(idxTrain,:);
            obj.y_Train = y(idxTrain,:);
            idxTest = test(cv);
            obj.X_Test = X(idxTest,:);
            obj.y_Test = y(idxTest,:);
        elseif contains(SplitMethod, 'SOH')
            %Use an SOH cutoff and grab that value
            maskSohCutoff = strcmp(varargin, 'SohCutoff');
            isSohCutoffSpecified = any(maskSohCutoff);
            
            %Apply the user specified SOH cutoff
            if isSohCutoffSpecified
                maskSohCutoffValue = logical([0 maskSohCutoff]);
                SohCutoff = varargin{maskSohCutoffValue};
                
                obj.X_Train = X(y>=SohCutoff);
                obj.X_Test = X(y<SohCutoff);
                obj.y_Train = y(y>=SohCutoff);
                obj.y_Test = y(y<SohCutoff);
            else
                disp("No 'SohCutoff' specified, so can't split data")
            end
        else
            disp('Current splitting method not supported')
        end
        
        %If no split specified we'll assume that the whole data set is used
        %for training.
        obj.X_Train = X; obj.y_Train = y;
    end

end