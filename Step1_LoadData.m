%%% Step1_LoadData
%%% Copyright (c) 2022 Alliance for Sustainable Energy, LLC
%%% Andrew Schiek 2022
%%% Script to load MIT Data Set and Create Linearized Voltage and Current
%%% curves.
%%% Data was originally published in "Data-driven prediction of battery
%%% cycle life before capacity degradation" by Severson et. al.
%%% Code from that paper can be found at: 
%%% https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

close all; clear all;

load('2017-05-12_batchdata_updated_struct_errorcorrect.mat')
soh = [];
IR = []; Tmax = []; Tavg = []; Tmin = [];
mxI = []; mnI = []; rngI = []; varI = [];
mxV = []; mnV = []; rngV = []; varV = [];
Vlin = []; Ilin = [];

for i = 1:length(batch)
    soh = [soh; batch(i).summary.QDischarge(2:end)/batch(i).summary.QDischarge(2)];
    IR = [IR; batch(i).summary.IR(2:end)];
    Tmax = [Tmax; batch(i).summary.Tmax(2:end)];
    Tavg = [Tavg; batch(i).summary.Tavg(2:end)];
    Tmin = [Tmin; batch(i).summary.Tmin(2:end)];
    
    temp_maxI = []; temp_minI = []; temp_rangeI = []; temp_vI = [];
    temp_maxV = []; temp_minV = []; temp_rangeV = []; temp_vV = [];
    temp_Vlin =[]; temp_Ilin = [];
    
    for k = 1:length(batch(i).cycles)-1
        maxI = max(batch(i).cycles(k+1).I); minI = min(batch(i).cycles(k+1).I);
        temp_maxI = [temp_maxI maxI]; temp_minI = [temp_minI minI]; temp_rangeI = [temp_rangeI maxI-minI]; 
        temp_vI = [temp_vI var(batch(i).cycles(k+1).I)];
        
        maxV = max(batch(i).cycles(k+1).V); minV = min(batch(i).cycles(k+1).V);
        temp_maxV = [temp_maxV maxV]; temp_minV = [temp_minV minV]; temp_rangeV = [temp_rangeV maxV-minV]; 
        temp_vV = [temp_vV var(batch(i).cycles(k+1).V)];
        
        temp_t = batch(i).cycles(k+1).t;
        temp_V = batch(i).cycles(k+1).V;
        temp_I = batch(i).cycles(k+1).I;
        temp_tt = linspace(min(temp_t), max(temp_t), 1000);
        
        try
            temp_Vlin = [temp_Vlin; spline(temp_t, temp_V, temp_tt)];
            temp_Ilin = [temp_Ilin; spline(temp_t, temp_I, temp_tt)];
        catch
            [temp_t, id, ~] = unique(temp_t, 'stable');
            temp_V = temp_V(id);
            temp_I = temp_I(id);
            temp_Vlin = [temp_Vlin; spline(temp_t, temp_V, temp_tt)];
            temp_Ilin = [temp_Ilin; spline(temp_t, temp_I, temp_tt)];
        end
    end
    mxI = [mxI; temp_maxI']; mnI = [mnI; temp_minI']; rngI = [rngI; temp_rangeI']; varI = [varI; temp_vI'];
    mxV = [mxV; temp_maxV']; mnV = [mnV; temp_minV']; rngV = [rngV; temp_rangeV']; varV = [varV; temp_vV'];
    Vlin = [Vlin; temp_Vlin]; Ilin = [Ilin; temp_Ilin];
end

X_MIT = table(IR, Tmax, Tavg, Tmin, mxI, mnI, rngI, varI, mxV, mnV, rngV, varV);
y_MIT = table(soh);
Vlin_MIT = Vlin;
Ilin_MIT = Ilin;
save('MIT_Variables.mat', 'X_MIT', 'y_MIT', 'Vlin_MIT', 'Ilin_MIT')
clearvars
