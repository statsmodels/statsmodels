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
Spec = load_spec('test_spec_blocks.xls');
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

k = Spec.k;
p = Spec.p;
kp = k * p;
p5 = max(5, Spec.p);
kp5 = k * p5;
Lambda = [Res.C(1:end, 1:k) ...
          Res.C(1:end, kp5 + 1:kp5 + k) ...
          Res.C(1:end, 2 * kp5 + 1:2 * kp5 + k)]';
mask = logical(Spec.Blocks');
Lambda = Lambda(mask)';

f_A1 = Res.A(1:k, 1:kp)';
f_A2 = Res.A(kp5 + 1:kp5 + k, kp5 + 1:kp5 + kp)';
f_A3 = Res.A(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + kp)';
f_A = [f_A1 f_A2 f_A3];

f_Q1 = Res.Q(1:k, 1:k);
f_Q2 = Res.Q(kp5 + 1:kp5 + k, kp5 + 1:kp5 + k);
f_Q3 = Res.Q(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + k);
f_Q = [nonzeros(chol(f_Q1))' ...
       nonzeros(chol(f_Q2))' ...
       nonzeros(chol(f_Q3))'];
   
Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = 3 * kp5 + 1;
params = [Lambda f_A f_Q ...
          Ad(idio:idio+6)' Ad(idio+6+5) Qd(idio:idio+6)' Qd(idio+6+5)]';
save('test_dfm_blocks_111.mat','Res','Spec','params');


%% Test 2: p=1, k=1, max_iter=1
Spec.p = 1;
Spec.k = 1;
max_iter = 2;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

k = Spec.k;
p = Spec.p;
kp = k * p;
p5 = max(5, Spec.p);
kp5 = k * p5;
Lambda = [Res.C(1:end, 1:k) ...
          Res.C(1:end, kp5 + 1:kp5 + k) ...
          Res.C(1:end, 2 * kp5 + 1:2 * kp5 + k)]';
mask = logical(Spec.Blocks');
Lambda = Lambda(mask)';

f_A1 = Res.A(1:k, 1:kp)';
f_A2 = Res.A(kp5 + 1:kp5 + k, kp5 + 1:kp5 + kp)';
f_A3 = Res.A(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + kp)';
f_A = [f_A1 f_A2 f_A3];

f_Q1 = Res.Q(1:k, 1:k);
f_Q2 = Res.Q(kp5 + 1:kp5 + k, kp5 + 1:kp5 + k);
f_Q3 = Res.Q(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + k);
f_Q = [nonzeros(chol(f_Q1))' ...
       nonzeros(chol(f_Q2))' ...
       nonzeros(chol(f_Q3))'];
   
Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = 3 * kp5 + 1;
params = [Lambda f_A f_Q ...
          Ad(idio:idio+6)' Ad(idio+6+5) Qd(idio:idio+6)' Qd(idio+6+5)]';
save('test_dfm_blocks_112.mat','Res','Spec','params');

%% Test 3: p=2, k=2, max_iter=1
Spec.p = 2;
Spec.k = 2;
max_iter = 1;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

k = Spec.k;
p = Spec.p;
kp = k * p;
p5 = max(5, Spec.p);
kp5 = k * p5;
Lambda1 = Res.C(1:end, 1:k);
mask1 = repmat(logical(Spec.Blocks(:, 1)), 1, 2);
Lambda2 = Res.C(1:end, kp5 + 1:kp5 + k);
mask2 = repmat(logical(Spec.Blocks(:, 2)), 1, 2);
Lambda3 = Res.C(1:end, 2 * kp5 + 1:2 * kp5 + k);
mask3 = repmat(logical(Spec.Blocks(:, 3)), 1, 2);
Lambda = [Lambda1 Lambda2 Lambda3]';
mask = [mask1 mask2 mask3]';
Lambda = Lambda(mask)';

f_A1 = Res.A(1:k, 1:kp)';
f_A2 = Res.A(kp5 + 1:kp5 + k, kp5 + 1:kp5 + kp)';
f_A3 = Res.A(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + kp)';
f_A = [f_A1 f_A2 f_A3];
f_A = f_A(:)';

f_Q1 = Res.Q(1:k, 1:k);
f_Q2 = Res.Q(kp5 + 1:kp5 + k, kp5 + 1:kp5 + k);
f_Q3 = Res.Q(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + k);
f_Q = [nonzeros(chol(f_Q1))' ...
       nonzeros(chol(f_Q2))' ...
       nonzeros(chol(f_Q3))'];
   
Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = 3 * kp5 + 1;
params = [Lambda f_A f_Q ...
          Ad(idio:idio+6)' Ad(idio+6+5) Qd(idio:idio+6)' Qd(idio+6+5)]';
save('test_dfm_blocks_221.mat','Res','Spec','params');

%% Test 3: p=2, k=2, max_iter=2
Spec.p = 2;
Spec.k = 2;
max_iter = 2;

Res = dfm(X,Spec,threshold,max_iter);
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});

k = Spec.k;
p = Spec.p;
kp = k * p;
p5 = max(5, Spec.p);
kp5 = k * p5;
Lambda1 = Res.C(1:end, 1:k);
mask1 = repmat(logical(Spec.Blocks(:, 1)), 1, 2);
Lambda2 = Res.C(1:end, kp5 + 1:kp5 + k);
mask2 = repmat(logical(Spec.Blocks(:, 2)), 1, 2);
Lambda3 = Res.C(1:end, 2 * kp5 + 1:2 * kp5 + k);
mask3 = repmat(logical(Spec.Blocks(:, 3)), 1, 2);
Lambda = [Lambda1 Lambda2 Lambda3]';
mask = [mask1 mask2 mask3]';
Lambda = Lambda(mask)';

f_A1 = Res.A(1:k, 1:kp)';
f_A2 = Res.A(kp5 + 1:kp5 + k, kp5 + 1:kp5 + kp)';
f_A3 = Res.A(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + kp)';
f_A = [f_A1 f_A2 f_A3];
f_A = f_A(:)';

f_Q1 = Res.Q(1:k, 1:k);
f_Q2 = Res.Q(kp5 + 1:kp5 + k, kp5 + 1:kp5 + k);
f_Q3 = Res.Q(2 * kp5 + 1:2 * kp5 + k, 2 * kp5 + 1:2 * kp5 + k);
f_Q = [nonzeros(chol(f_Q1))' ...
       nonzeros(chol(f_Q2))' ...
       nonzeros(chol(f_Q3))'];
   
Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = 3 * kp5 + 1;
params = [Lambda f_A f_Q ...
          Ad(idio:idio+6)' Ad(idio+6+5) Qd(idio:idio+6)' Qd(idio+6+5)]';
save('test_dfm_blocks_222.mat','Res','Spec','params');
