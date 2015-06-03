% Load the dataset
data = csvread('results_wpi1_ar3_stata.csv', 1, 0);

% First differences
dwpi1 = diff(data(:,1));

% Parameters
ar = [.5270715, .0952613, .2580355];
sigma2 = .5307459;

% Create the state space representation
[nobs, k_endog] = size(dwpi1);
k_states = 3;
k_posdef = 1;

obs = dwpi1';
obs_intercept = [0];
design = [1, 0, 0];
obs_cov = [0];

state_intercept = [0; 0];
transition = [ar; 1, 0, 0; 0, 1, 0];
selection = [1; 0; 0];
state_cov = [sigma2];
selected_state_cov = selection * state_cov * selection';

initial_state = [0; 0];
initial_state_cov = dlyap(conj(transition), selected_state_cov);

mod = ssmodel('test', ssmat(obs_cov), ssmat(design), ssmat(transition),ssmat(selection),ssmat(state_cov));
mod.P1 = initial_state_cov;

% Optionally add missing values
% obs(10:20) = nan;

% Estimate
[a, P] = kalman(dwpi1', mod);
[alphahat, V] = statesmo(dwpi1', mod);
[eps, eta, epsvar, etavar] = disturbsmo(dwpi1', mod);
% Note: simsmo seems to always crashes MATLAB
%[alphatilde, epstilde, etatilde] = simsmo(dwpi1', mod);

% Calculate determinants of variance matrices (so that we can compare
% in unit tests)
detP = zeros(1,nobs);
detV = zeros(1,nobs);
for i = 1:nobs+1;
    detP(i) = det(P(:,:,i));
    detV(i) = det(V(:,:,i));
end;

% Write output
csvwrite('results_wpi1_ar3_matlab_ssm.csv', [a(:,1:end-1) detP(:,1:end-1) alphahat detV eps epsvar eta etavar]);
