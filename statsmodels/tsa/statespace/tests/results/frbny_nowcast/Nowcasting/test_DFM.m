%%% Test for Dynamic factor model (DFM) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script produces results for a variety of dynamic factor models (DFM)
% using a panel of monthly and quarterly series that can be used for tests.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Clear workspace and set paths.
close all; clear; clc;
addpath('functions');


%% User inputs.
vintage = '2016-06-29'; % vintage dataset to use for estimation
country = 'US';         % United States macroeconomic data
sample_start  = datenum('2000-01-01','yyyy-mm-dd'); % estimation sample


%% Load model specification and dataset.
% Load model specification structure `Spec`
Spec = load_spec('test_spec.xls');
% Load data
datafile = fullfile('data',country,[vintage '.xls']);
[X,Time,Z] = load_data(datafile,Spec,sample_start);

% Basic settings
threshold = 1e-4; % Set to 1e-5 for more robust estimates
format long

%% Test 1: p=1, k=1, max_iter=1
Spec.p = 1;
Spec.k = 1;
max_iter = 1;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';
save('test_dfm_111.mat','Res','Spec','params');


%% Test 2: p=1, k=1, max_iter=2
Spec.p = 1;
Spec.k = 1;
max_iter = 2;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';
save('test_dfm_112.mat','Res','Spec','params');


%% Test 3: p=1, k=1, max_iter=5000
Spec.p = 1;
Spec.k = 1;
max_iter = 5000;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';
save('test_dfm_11F.mat','Res','Spec','params');

%% Test 4: p=2, k=2, max_iter=1
Spec.p = 2;
Spec.k = 2;
max_iter = 1;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';
save('test_dfm_221.mat','Res','Spec','params');


%% Test 5: p=2, k=2, max_iter=2
Spec.p = 2;
Spec.k = 2;
max_iter = 2;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';
save('test_dfm_222.mat','Res','Spec','params');


%% Test 6: p=2, k=2, max_iter=5000
Spec.p = 2;
Spec.k = 2;
max_iter = 5000;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';
save('test_dfm_22F.mat','Res','Spec','params');
