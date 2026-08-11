%%% Dynamic factor model (DFM) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script estimates a dynamic factor model (DFM) using a panel of
% monthly and quarterly series.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Clear workspace and set paths.
close all; clear; clc;
addpath('functions');


%% User inputs.
vintage_old = '2016-06-29'; % vintage dataset to use for estimation
vintage_new = '2016-07-29'; % updated vintage dataset
country = 'US';             % United States macroeconomic data
sample_start  = datenum('2000-01-01','yyyy-mm-dd'); % estimation sample

series = 'GDPC1' ; % Nowcasting real GDP (GDPC1) <fred.stlouisfed.org/series/GDPC1>
period = '2016q3'; % Forecasting target quarter


%% Load model specification and dataset.
Spec = load_spec('test_spec.xls');
datafile_old = fullfile('data', country, [vintage_old '.xls']);
datafile_new = fullfile('data', country, [vintage_new '.xls']);
[X_old, ~] = load_data(datafile_old, Spec, sample_start);
[X_new, Time] = load_data(datafile_new, Spec, sample_start);

% Basic settings
threshold = 1e-4; % Set to 1e-5 for more robust estimates
format long

%% Test 1: p=1, k=1, max_iter=2
Spec.p = 1;
Spec.k = 1;
max_iter = 2;

% Run DFM model
Res = dfm(X_old, Spec, threshold, max_iter);

% Update nowcast
Res = update_nowcast(X_old, X_new, Time, Spec, Res, series, period, ...
                     vintage_old, vintage_new);


% Store results
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});
Res.news_table = Res.news_table{:,:};

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';

save('test_news_112.mat','Res','Spec','params');


%% Test 2: p=2, k=2, max_iter=2
Spec.p = 2;
Spec.k = 2;
max_iter = 2;

% Run DFM model
Res = dfm(X_old, Spec, threshold, max_iter);

% Update nowcast
Res = update_nowcast(X_old, X_new, Time, Spec, Res, series, period, ...
                     vintage_old, vintage_new);


% Store results
Res = rmfield(Res, {'X_sm', 'Z', 'Vsmooth', 'VVsmooth'});
Res.news_table = Res.news_table{:,:};

kp = Spec.k * Spec.p;
Lambda = Res.C(1:end, 1:Spec.k)';
f_A = Res.A(1:Spec.k, 1:kp)';
f_Q = Res.Q(1:Spec.k, 1:Spec.k);

Ad = diag(Res.A);
Qd = diag(Res.Q);

idio = Spec.k * max(5, Spec.p) + 1;
params = [Lambda(:)' f_A(:)' nonzeros(chol(f_Q)')' ...
          Ad(idio:idio+3)' Ad(idio+3+5) Qd(idio:idio+3)' Qd(idio+3+5)]';

save('test_news_222.mat','Res','Spec','params');
