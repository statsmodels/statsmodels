function Res = dfm(X,Spec,threshold,max_iter)
%DFM()    Runs the dynamic factor model
%
%  Syntax:
%    Res = DFM(X,Par)
%
%  Description:
%   DFM() inputs the organized and transformed data X and parameter structure Par.
%   Then, the function outputs dynamic factor model structure Res and data
%   summary statistics (mean and standard deviation).
%
%  Input arguments:
%    X: Kalman-smoothed data where missing values are replaced by their expectation
%    Par: A structure containing the following parameters:
%      Par.blocks: Block loadings.
%      Par.nQ: Number of quarterly series
%      Par.p: Number of lags in transition matrix
%      Par.r: Number of common factors for each block
%
% Output Arguments:
%
%   Res - structure of model results with the following fields
%       . X_sm | Kalman-smoothed data where missing values are replaced by their expectation
%       . Z | Smoothed states. Rows give time, and columns are organized according to Res.C.
%       . C | Observation matrix. The rows correspond
%          to each series, and the columns are organized as shown below:
%         - 1-20: These columns give the factor loadings. For example, 1-5
%              give loadings for the first block and are organized in
%              reverse-chronological order (f^G_t, f^G_t-1, f^G_t-2, f^G_t-3,
%              f^G_t-4). Columns 6-10, 11-15, and 16-20 give loadings for
%              the second, third, and fourth blocks respectively.
%       .R: Covariance for observation matrix residuals
%       .A: Transition matrix. This is a square matrix that follows the
%      same organization scheme as Res.C's columns. Identity matrices are
%      used to account for matching terms on the left and righthand side.
%      For example, we place an I4 matrix to account for matching
%      (f_t-1; f_t-2; f_t-3; f_t-4) terms.
%       .Q: Covariance for transition equation residuals.
%       .Mx: Series mean
%       .Wx: Series standard deviation
%       .Z_0: Initial value of state
%       .V_0: Initial value of covariance matrix
%       .r: Number of common factors for each block
%       .p: Number of lags in transition equation
%
% References:
%
%   Marta Banbura, Domenico Giannone and Lucrezia Reichlin
%   Nowcasting (2010)
%   Michael P. Clements and David F. Hendry, editors,
%   Oxford Handbook on Economic Forecasting.

%% Store model parameters ------------------------------------------------


% DFM input specifications: See documentation for details
Par.blocks = Spec.Blocks;                  % Block loading structure
Par.nQ = sum(strcmp('q',Spec.Frequency));  % Number of quarterly series
Par.p = Spec.p;                                 % Number of lags in autoregressive of factor (same for all factors)
Par.r = ones(1,size(Spec.Blocks,2)) * Spec.k;       % Number of common factors for each block

%Par.r(1) =2;
% Display blocks
try
    fprintf('\n\n\n');
    disp('Table 3: Block Loading Structure')
    disp(array2table(Spec.Blocks,...
         'RowNames', strrep(Spec.SeriesName,' ','_'),...
         'VariableNames',Spec.BlockNames));
    fprintf('\n\n\n');
catch
end

fprintf('Estimating the dynamic factor model (DFM) ... \n\n');

[T,N] = size(X);
r = Par.r;
p = Par.p;
nQ = Par.nQ;
blocks = Par.blocks;

i_idio = logical([ones(N-nQ,1);zeros(nQ,1)]);

%R*Lambda = q; Constraints on the loadings of the quartrly variables

R_mat = [  2 -1  0  0  0;...
           3  0 -1  0  0;...
           2  0  0 -1  0;...
           1  0  0  0 -1];

q = zeros(4,1);

if(nargin < 3)
    threshold = 1e-5;  % EM loop threshold (default value)
end

if(nargin < 4)
    max_iter = 5000;  % EM loop maximum number of iterations
end

%% Prepare data -----------------------------------------------------------

Mx = mean(X,'omitnan');
Wx = std(X,'omitnan');
xNaN = (X-repmat(Mx,T,1))./repmat(Wx,T,1);  % Standardize series

%% Initial Conditions -----------------------------------------------------

optNaN.method = 2; % Remove leading and closing zeros
optNaN.k = 3;      % Setting for filter(): See remNaN_spline

[A, C, Q, R, Z_0, V_0] = InitCond(xNaN,r,p,blocks,optNaN,R_mat,q,nQ,i_idio);

% Initialize EM loop values
previous_loglik = -inf;
num_iter = 0;
LL = -inf;
converged = 0;

% y for the estimation is WITH missing data
y = xNaN';

%% EM LOOP ----------------------------------------------------------------

%The model can be written as
%y = C*Z + e;
%Z = A*Z(-1) + v
%where y is NxT, Z is (pr)xT, etc

% Remove the leading and ending nans
optNaN.method = 3;
y_est = remNaNs_spline(xNaN,optNaN)';

while (num_iter < max_iter) & ~converged % Loop until converges or max iter.

    [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik] = ...  % Applying EM algorithm
        EMstep(y_est, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ,i_idio,blocks);

    C = C_new;
    R = R_new;
    A = A_new;
    Q = Q_new;

    if num_iter > 2  % Checking convergence
        [converged, decrease(num_iter + 1)] = ...
            em_converged(loglik, previous_loglik, threshold, 1);
    end

    if (mod(num_iter,10) == 0) && (num_iter > 0)  % Print updates to command window
        disp(['Now running the ',num2str(num_iter),...
              'th iteration of max ', num2str(max_iter)]);
        disp(['  Loglik','   (% Change)'])
        disp([num2str(loglik),'   (', sprintf('%6.2f',100*((loglik-previous_loglik)/previous_loglik)) '%)'])
    end


    LL = [LL loglik];
    previous_loglik = loglik;
    num_iter =  num_iter + 1;

end

if(num_iter < max_iter)
    disp(['Successful: Convergence at ', num2str(num_iter), ' iterations'])
else
   disp('Stopped because maximum iterations reached')
end

% Final run of the Kalman filter
[Zsmooth, Vsmooth, VVsmooth, loglik] = runKF(y, A, C, Q, R, Z_0, V_0);
Zsmooth = Zsmooth';

x_sm = Zsmooth(2:end,:) * C';  % Get smoothed X


%%  Loading the structure with the results --------------------------------
Res.x_sm = x_sm;
Res.X_sm = repmat(Wx,T,1) .* x_sm + repmat(Mx,T,1);  % Unstandardized, smoothed
Res.Z = Zsmooth(2:end,:);
Res.C = C;
Res.R = R;
Res.A = A;
Res.Q = Q;
Res.Mx = Mx;
Res.Wx = Wx;
Res.Z_0 = Z_0;
Res.V_0 = V_0;
Res.r = r;
Res.p = p;
Res.loglik = loglik;
Res.Vsmooth = Vsmooth;
Res.VVsmooth = VVsmooth;

%% Display output
% Table with names and factor loadings

nQ       = Par.nQ;                      % Number of quarterly series
nM       = size(Spec.SeriesID,1) - nQ;  % Number monthly series
nLags    = max(Par.p, 5);               % 5 comes from monthly-quarterly aggregation
nFactors = sum(Par.r);

fprintf('\n\n\n');

try
disp('Table 4: Factor Loadings for Monthly Series');
disp(array2table(Res.C(1:nM, 1:5:nFactors*5),...  % Only select lag(0) terms
     'RowNames', strrep(Spec.SeriesName(1:nM),' ','_'), ...
     'VariableNames', Spec.BlockNames));
fprintf('\n\n\n');
disp('Table 5: Quarterly Loadings Sample (Global Factor)')
disp(array2table(Res.C(end-nQ+1:end, 1:5), ...  % Select only quarterly series
     'RowNames', strrep(Spec.SeriesName(end-nQ+1:end),' ','_'), ...
     'VariableNames', {'f1_lag0', 'f1_lag1', 'f1_lag2', 'f1_lag3', 'f1_lag4'}));
fprintf('\n\n\n');
catch
end

% Table with AR model on factors (factors with AR parameter and variance of residuals)

A_terms = diag(Res.A);  % Transition equation terms
Q_terms = diag(Res.Q);  % Covariance matrix terms

try
disp('Table 6: Autoregressive Coefficients on Factors')
disp(table(A_terms(1:5:nFactors*5), ...  % Only select lag(0) terms
           Q_terms(1:5:nFactors*5), ...
           'VariableNames', {'AR_Coefficient', 'Variance_Residual'}, ...
           'RowNames',      strrep(Spec.BlockNames,' ','_')));
fprintf('\n\n\n');
catch
end

% Table with AR model idiosyncratic errors (factors with AR parameter and variance of residuals)
try
disp('Table 7: Autoregressive Coefficients on Idiosyncratic Component')
disp(table(A_terms([nFactors*5+1:nFactors*5+nM nFactors*5+nM+1:5:end]),...  % 21:50 give monthly, 51:5:61 give quarterly
           Q_terms([nFactors*5+1:nFactors*5+nM nFactors*5+nM+1:5:end]), ...
           'VariableNames', {'AR_Coefficient', 'Variance_Residual'}, ...
           'RowNames', strrep(Spec.SeriesName,' ','_')));
catch
end

end



%% PROCEDURES -------------------------------------------------------------
% Note: Kalman filter (runKF()) is in the 'functions' folder

function  [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik] = EMstep(y, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ,i_idio,blocks)
%EMstep    Applies EM algorithm for parameter reestimation
%
%  Syntax:
%    [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik]
%    = EMstep(y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks)
%
%  Description:
%    EMstep reestimates parameters based on the Estimation Maximization (EM)
%    algorithm. This is a two-step procedure:
%    (1) E-step: the expectation of the log-likelihood is calculated using
%        previous parameter estimates.
%    (2) M-step: Parameters are re-estimated through the maximisation of
%        the log-likelihood (maximize result from (1)).
%
%    See "Maximum likelihood estimation of factor models on data sets with
%    arbitrary pattern of missing data" for details about parameter
%    derivation (Banbura & Modugno, 2010). This procedure is in much the
%    same spirit.
%
%  Input:
%    y:      Series data
%    A:      Transition matrix
%    C:      Observation matrix
%    Q:      Covariance for transition equation residuals
%    R:      Covariance for observation matrix residuals
%    Z_0:    Initial values of factors
%    V_0:    Initial value of factor covariance matrix
%    r:      Number of common factors for each block (e.g. vector [1 1 1 1])
%    p:      Number of lags in transition equation
%    R_mat:  Estimation structure for quarterly variables (i.e. "tent")
%    q:      Constraints on loadings
%    nQ:     Number of quarterly series
%    i_idio: Indices for monthly variables
%    blocks: Block structure for each series (i.e. for a series, the structure
%            [1 0 0 1] indicates loadings on the first and fourth factors)
%
%  Output:
%    C_new: Updated observation matrix
%    R_new: Updated covariance matrix for residuals of observation matrix
%    A_new: Updated transition matrix
%    Q_new: Updated covariance matrix for residuals for transition matrix
%    Z_0:   Initial value of state
%    V_0:   Initial value of covariance matrix
%    loglik: Log likelihood
%
% References:
%   "Maximum likelihood estimation of factor models on data sets with
%   arbitrary pattern of missing data" by Banbura & Modugno (2010).
%   Abbreviated as BM2010
%
%

%% Initialize preliminary values

% Store series/model values
[n, T] = size(y);
nM = n - nQ;  % Number of monthly series
pC = size(R_mat,2);
ppC = max(p,pC);
num_blocks = size(blocks,2);  % Number of blocks

%% ESTIMATION STEP: Compute the (expected) sufficient statistics for a single
%Kalman filter sequence

% Running the Kalman filter and smoother with current parameters
% Note that log-liklihood is NOT re-estimated after the runKF step: This
% effectively gives the previous iteration's log-likelihood
% For more information on output, see runKF
[Zsmooth, Vsmooth, VVsmooth, loglik] = runKF(y, A, C, Q, R, Z_0, V_0);


%% MAXIMIZATION STEP (TRANSITION EQUATION)
% See (Banbura & Modugno, 2010) for details.

% Initialize output
A_new = A;
Q_new = Q;
V_0_new = V_0;

%%% 2A. UPDATE FACTOR PARAMETERS INDIVIDUALLY ----------------------------

for i = 1:num_blocks  % Loop for each block: factors are uncorrelated

    % SETUP INDEXING
    r_i = r(i);  % r_i = 1 if block is loaded
    rp = r_i*p;
    rp1 = sum(r(1:i-1))*ppC;
    b_subset = rp1+1:rp1+rp;  % Subset blocks: Helps for subsetting Zsmooth, Vsmooth
    t_start = rp1+1;          % Transition matrix factor idx start
    t_end = rp1+r_i*ppC;      % Transition matrix factor idx end



    % ESTIMATE FACTOR PORTION OF Q, A
    % Note: EZZ, EZZ_BB, EZZ_FB are parts of equations 6 and 8 in BM 2010

    % E[f_t*f_t' | Omega_T]
    EZZ = Zsmooth(b_subset, 2:end) * Zsmooth(b_subset, 2:end)'...
        +sum(Vsmooth(b_subset, b_subset, 2:end) ,3);

    % E[f_{t-1}*f_{t-1}' | Omega_T]
    EZZ_BB = Zsmooth(b_subset, 1:end-1)*Zsmooth(b_subset, 1:end-1)'...
            +sum(Vsmooth(b_subset, b_subset, 1:end-1), 3);

    % E[f_t*f_{t-1}' | Omega_T]
    EZZ_FB = Zsmooth(b_subset, 2:end)*Zsmooth(b_subset, 1:end-1)'...
        +sum(VVsmooth(b_subset, b_subset, :), 3);

    % Select transition matrix/covariance matrix for block i
    A_i = A(t_start:t_end, t_start:t_end);
    Q_i = Q(t_start:t_end, t_start:t_end);

    % Equation 6: Estimate VAR(p) for factor
    A_i(1:r_i,1:rp) = EZZ_FB(1:r_i,1:rp) * inv(EZZ_BB(1:rp,1:rp));

    % Equation 8: Covariance matrix of residuals of VAR
    Q_i(1:r_i,1:r_i) = (EZZ(1:r_i,1:r_i) - A_i(1:r_i,1:rp)* EZZ_FB(1:r_i,1:rp)') / T;

    % Place updated results in output matrix
    A_new(t_start:t_end, t_start:t_end) = A_i;
    Q_new(t_start:t_end, t_start:t_end) = Q_i;
    V_0_new(t_start:t_end, t_start:t_end) =...
        Vsmooth(t_start:t_end, t_start:t_end,1);
end

%%% 2B. UPDATING PARAMETERS FOR IDIOSYNCRATIC COMPONENT ------------------

rp1 = sum(r)*ppC;           % Col size of factor portion
niM = sum(i_idio(1:nM));    % Number of monthly values
t_start = rp1+1;            % Start of idiosyncratic component index
i_subset = t_start:rp1+niM; % Gives indices for monthly idiosyncratic component values


% Below 3 estimate the idiosyncratic component (for eqns 6, 8 BM 2010)

% E[f_t*f_t' | \Omega_T]
EZZ = diag(diag(Zsmooth(t_start:end, 2:end) * Zsmooth(t_start:end, 2:end)'))...
    + diag(diag(sum(Vsmooth(t_start:end, t_start:end, 2:end), 3)));

% E[f_{t-1}*f_{t-1}' | \Omega_T]
EZZ_BB = diag(diag(Zsmooth(t_start:end, 1:end-1)* Zsmooth(t_start:end, 1:end-1)'))...
       + diag(diag(sum(Vsmooth(t_start:end, t_start:end, 1:end-1), 3)));

% E[f_t*f_{t-1}' | \Omega_T]
EZZ_FB = diag(diag(Zsmooth(t_start:end, 2:end)*Zsmooth(t_start:end, 1:end-1)'))...
       + diag(diag(sum(VVsmooth(t_start:end, t_start:end, :), 3)));

A_i = EZZ_FB * diag(1./diag((EZZ_BB)));  % Equation 6
Q_i = (EZZ - A_i*EZZ_FB') / T;           % Equation 8

% Place updated results in output matrix
A_new(i_subset, i_subset) = A_i(1:niM,1:niM);
Q_new(i_subset, i_subset) = Q_i(1:niM,1:niM);
V_0_new(i_subset, i_subset) = diag(diag(Vsmooth(i_subset, i_subset, 1)));


%% 3 MAXIMIZATION STEP (observation equation)

%%% INITIALIZATION AND SETUP ----------------------------------------------
Z_0 = Zsmooth(:,1); %zeros(size(Zsmooth,1),1); %
V_0 = Vsmooth(:,:,1);

% Set missing data series values to 0
nanY = isnan(y);
y(nanY) = 0;

% LOADINGS
C_new = C;

% Blocks
bl = unique(blocks,'rows');  % Gives unique loadings
n_bl = size(bl,1);           % Number of unique loadings

% Initialize indices: These later help with subsetting
bl_idxM = [];  % Indicator for monthly factor loadings
bl_idxQ = [];  % Indicator for quarterly factor loadings
R_con = [];    % Block diagonal matrix giving monthly-quarterly aggreg scheme
q_con = [];

% Loop through each block
for i = 1:num_blocks
    bl_idxQ = [bl_idxQ repmat(bl(:,i),1,r(i)*ppC)];
    bl_idxM = [bl_idxM repmat(bl(:,i),1,r(i)) zeros(n_bl,r(i)*(ppC-1))];
    R_con = blkdiag(R_con, kron(R_mat,eye(r(i))));
    q_con = [q_con;zeros(r(i)*size(R_mat,1),1)];
end

% Indicator for monthly/quarterly blocks in observation matrix
bl_idxM = logical(bl_idxM);
bl_idxQ = logical(bl_idxQ);

i_idio_M = i_idio(1:nM);            % Gives 1 for monthly series
n_idio_M = length(find(i_idio_M));  % Number of monthly series
c_i_idio = cumsum(i_idio);          % Cumulative number of monthly series

for i = 1:n_bl  % Loop through unique loadings (e.g. [1 0 0 0], [1 1 0 0])

    bl_i = bl(i,:);
    rs = sum(r(logical(bl_i)));                    % Total num of blocks loaded
    idx_i = find(ismember(blocks, bl_i, 'rows'));  % Indices for bl_i
    idx_iM = idx_i(idx_i<nM+1);                    % Only monthly
    n_i = length(idx_iM);                          % Number of monthly series

    % Initialize sums in equation 13 of BGR 2010
    denom = zeros(n_i*rs,n_i*rs);
    nom = zeros(n_i,rs);

    % Stores monthly indicies. These are done for input robustness
    i_idio_i = i_idio_M(idx_iM);
    i_idio_ii = c_i_idio(idx_iM);
    i_idio_ii = i_idio_ii(i_idio_i);

    %%% UPDATE MONTHLY VARIABLES: Loop through each period ----------------
    for t = 1:T
        Wt = diag(~nanY(idx_iM, t));  % Gives selection matrix (1 for nonmissing values)

        denom = denom +...  % E[f_t*t_t' | Omega_T]
                kron(Zsmooth(bl_idxM(i, :), t+1) * Zsmooth(bl_idxM(i, :), t+1)' + ...
                Vsmooth(bl_idxM(i, :), bl_idxM(i, :), t+1), Wt);

        nom = nom + ...  E[y_t*f_t' | \Omega_T]
              y(idx_iM, t) * Zsmooth(bl_idxM(i, :), t+1)' - ...
              Wt(:, i_idio_i) * (Zsmooth(rp1 + i_idio_ii, t+1) * ...
              Zsmooth(bl_idxM(i, :), t+1)' + ...
              Vsmooth(rp1 + i_idio_ii, bl_idxM(i, :), t+1));
    end

    vec_C = inv(denom)*nom(:);  % Eqn 13 BGR 2010

    % Place updated monthly results in output matrix
    C_new(idx_iM,bl_idxM(i,:)) = reshape(vec_C, n_i, rs);

   %%% UPDATE QUARTERLY VARIABLES -----------------------------------------

   idx_iQ = idx_i(idx_i > nM);  % Index for quarterly series
   rps = rs * ppC;

   % Monthly-quarterly aggregation scheme
   R_con_i = R_con(:,bl_idxQ(i,:));
   q_con_i = q_con;

   no_c = ~(any(R_con_i,2));
   R_con_i(no_c,:) = [];
   q_con_i(no_c,:) = [];

   % Loop through quarterly series in loading. This parallels monthly code
   for j = idx_iQ'
       % Initialization
       denom = zeros(rps,rps);
       nom = zeros(1,rps);
       nom2 = zeros(1,rps);

       idx_jQ = j-nM;  % Ordinal position of quarterly variable
       % Loc of factor structure corresponding to quarterly var residuals
       i_idio_jQ = (rp1 + n_idio_M + 5*(idx_jQ-1)+1:rp1+ n_idio_M + 5*idx_jQ);

       % Place quarterly values in output matrix
       V_0_new(i_idio_jQ, i_idio_jQ) = Vsmooth(i_idio_jQ, i_idio_jQ,1);
       A_new(i_idio_jQ(1), i_idio_jQ(1)) = A_i(i_idio_jQ(1)-rp1, i_idio_jQ(1)-rp1);
       Q_new(i_idio_jQ(1), i_idio_jQ(1)) = Q_i(i_idio_jQ(1)-rp1, i_idio_jQ(1)-rp1);

       for t=1:T
           Wt = diag(~nanY(j,t));  % Selection matrix for quarterly values

           % Intermediate steps in BGR equation 13
           denom = denom + ...
                   kron(Zsmooth(bl_idxQ(i,:), t+1) * Zsmooth(bl_idxQ(i,:), t+1)'...
                 + Vsmooth(bl_idxQ(i,:), bl_idxQ(i,:), t+1), Wt);
           nom = nom + y(j, t)*Zsmooth(bl_idxQ(i,:), t+1)';
           nom = nom -...
                Wt * ([1 2 3 2 1] * Zsmooth(i_idio_jQ,t+1) * ...
                Zsmooth(bl_idxQ(i,:),t+1)'+...
                [1 2 3 2 1]*Vsmooth(i_idio_jQ,bl_idxQ(i,:),t+1));
           nom2 = nom2 + Wt * ([1 2 3 2 1] * Zsmooth(i_idio_jQ,t+1) * ...
                Zsmooth(bl_idxQ(i,:),t+1)'+...
                [1 2 3 2 1] * Vsmooth(i_idio_jQ,bl_idxQ(i,:),t+1));
       end

        C_i = inv(denom) * nom';
        C_i_constr = C_i - ...  % BGR equation 13
                     inv(denom) * R_con_i'*inv(R_con_i*inv(denom)*R_con_i') * (R_con_i*C_i-q_con_i);

        % Place updated values in output structure
        C_new(j,bl_idxQ(i,:)) = C_i_constr;
   end
end

%%% 3B. UPDATE COVARIANCE OF RESIDUALS FOR OBSERVATION EQUATION -----------
% Initialize covariance of residuals of observation equation
R_new = zeros(n,n);
for t=1:T
    Wt = diag(~nanY(:,t));  % Selection matrix
    R_new = R_new + (y(:,t) - ...  % BGR equation 15
            Wt * C_new * Zsmooth(:, t+1)) * (y(:,t) - Wt*C_new*Zsmooth(:,t+1))'...
           + Wt*C_new*Vsmooth(:,:,t+1)*C_new'*Wt + (eye(n)-Wt)*R*(eye(n)-Wt);
end


R_new = R_new/T;
RR = diag(R_new); %RR(RR<1e-2) = 1e-2;
RR(i_idio_M) = 1e-04;  % Ensure non-zero measurement error. See Doz, Giannone, Reichlin (2012) for reference.
RR(nM+1:end) = 1e-04;
R_new = diag(RR);

end



%--------------------------------------------------------------------------

function [converged, decrease] = em_converged(loglik, previous_loglik, threshold, check_decreased)
%em_converged    checks whether EM algorithm has converged
%
%  Syntax:
%    [converged, decrease] = em_converged(loglik, previous_loglik, threshold, check_increased)
%
%  Description:
%    em_converged() checks whether EM has converged. Convergence occurs if
%    the slope of the log-likelihood function falls below 'threshold'(i.e.
%    f(t) - f(t-1)| / avg < threshold) where avg = (|f(t)| + |f(t-1)|)/2
%    and f(t) is log lik at iteration t. 'threshold' defaults to 1e-4.
%
%    This stopping criterion is from Numerical Recipes in C (pg. 423).
%    With MAP estimation (using priors), the likelihood can decrease
%    even if the mode of the posterior increases.
%
%  Input arguments:
%    loglik: Log-likelihood from current EM iteration
%    previous_loglik: Log-likelihood from previous EM iteration
%    threshold: Convergence threshhold. The default is 1e-4.
%    check_decreased: Returns text output if log-likelihood decreases.
%
%  Output:
%    converged (numeric): Returns 1 if convergence criteria satisfied, and 0 otherwise.
%    decrease (numeric): Returns 1 if loglikelihood decreased.

%% Instantiate variables

% Threshhold arguments: Checks default behavior
if nargin < 3, threshold = 1e-4; end
if nargin < 4, check_decreased = 1; end

% Initialize output
converged = 0;
decrease = 0;

%% Check if log-likelihood decreases (optional)

if check_decreased
    if loglik - previous_loglik < -1e-3 % allow for a little imprecision
        fprintf(1, '******likelihood decreased from %6.4f to %6.4f!\n', previous_loglik, loglik);
        decrease = 1;
    end
end

%% Check convergence criteria

delta_loglik = abs(loglik - previous_loglik);  % Difference in loglik
avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2;

if (delta_loglik / avg_loglik) < threshold,
    converged = 1;  % Check convergence
end

end

%--------------------------------------------------------------------------

%InitCond()      Calculates initial conditions for parameter estimation
%
%  Description:
%    Given standardized data and model information, InitCond() creates
%    initial parameter estimates. These are intial inputs in the EM
%    algorithm, which re-estimates these parameters using Kalman filtering
%    techniques.
%
%Inputs:
%  - x:      Standardized data
%  - r:      Number of common factors for each block
%  - p:      Number of lags in transition equation
%  - blocks: Gives series loadings
%  - optNaN: Option for missing values in spline. See remNaNs_spline() for details.
%  - Rcon:   Incorporates estimation for quarterly series (i.e. "tent structure")
%  - q:      Constraints on loadings for quarterly variables
%  - NQ:     Number of quarterly variables
%  - i_idio: Logical. Gives index for monthly variables (1) and quarterly (0)
%
%Output:
%  - A:   Transition matrix
%  - C:   Observation matrix
%  - Q:   Covariance for transition equation residuals
%  - R:   Covariance for observation equation residuals
%  - Z_0: Initial value of state
%  - V_0: Initial value of covariance matrix

function [ A, C, Q, R, Z_0, V_0] = InitCond(x,r,p,blocks,optNaN,Rcon,q,nQ,i_idio)

pC = size(Rcon,2);  % Gives 'tent' structure size (quarterly to monthly)
ppC = max(p,pC);
n_b = size(blocks,2);  % Number of blocks

OPTS.disp=0;  % Turns off diagnostic information for eigenvalue computation
[xBal,indNaN] = remNaNs_spline(x,optNaN);  % Spline without NaNs

[T,N] = size(xBal);  % Time T series number N
nM = N-nQ;           % Number of monthly series

xNaN = xBal;
xNaN(indNaN) = nan;  % Set missing values equal to NaNs
res = xBal;          % Spline output equal to res: Later this is used for residuals
resNaN = xNaN;       % Later used for residuals

% Initialize model coefficient output
C = [];
A = [];
Q = [];
V_0 = [];

% Set the first observations as NaNs: For quarterly-monthly aggreg. scheme
indNaN(1:pC-1,:) = true;

for i = 1:n_b  % Loop for each block

    r_i = r(i);  % r_i = 1 when block is loaded

    %% Observation equation -----------------------------------------------

    C_i = zeros(N,r_i*ppC);     % Initialize state variable matrix helper
    idx_i = find(blocks(:,i));  % Returns series index loading block i
    idx_iM = idx_i(idx_i<nM+1); % Monthly series indicies for loaded blocks
    idx_iQ = idx_i(idx_i>nM);   % Quarterly series indicies for loaded blocks



    % Returns eigenvector v w/largest eigenvalue d
    [v, d] = eigs(cov(res(:,idx_iM)), r_i, 'lm');

    % Flip sign for cleaner output. Gives equivalent results without this section
    if(sum(v) < 0)
        v = -v;
    end

    % For monthly series with loaded blocks (rows), replace with eigenvector
    % This gives the loading
    C_i(idx_iM,1:r_i) = v;
    f = res(:,idx_iM)*v;  % Data projection for eigenvector direction
    F = [];

    % Lag matrix using loading. This is later used for quarterly series
    for kk = 0:max(p+1,pC)-1
        F = [F f(pC-kk:end-kk,:)];
    end

    Rcon_i = kron(Rcon,eye(r_i));  % Quarterly-monthly aggregation scheme
    q_i = kron(q,zeros(r_i,1));

    % Produces projected data with lag structure (so pC-1 fewer entries)
    ff = F(:, 1:r_i*pC);

    for j = idx_iQ'      % Loop for quarterly variables

        % For series j, values are dropped to accommodate lag structure
        xx_j = resNaN(pC:end,j);

        if sum(~isnan(xx_j)) < size(ff,2)+2
            xx_j = res(pC:end,j);  % Replaces xx_j with spline if too many NaNs

        end

        ff_j = ff(~isnan(xx_j),:);
        xx_j = xx_j(~isnan(xx_j));

        iff_j = inv(ff_j'*ff_j);
        Cc = iff_j*ff_j'*xx_j;  % Least squares

        % Spline data monthly to quarterly conversion
        Cc = Cc - iff_j*Rcon_i'*inv(Rcon_i*iff_j*Rcon_i')*(Rcon_i*Cc-q_i);

        C_i(j,1:pC*r_i)=Cc';  % Place in output matrix
    end

    ff = [zeros(pC-1,pC*r_i);ff];  % Zeros in first pC-1 entries (replace dropped from lag)

    % Residual calculations
    res = res - ff*C_i';
    resNaN = res;
    resNaN(indNaN) = nan;

    C = [C C_i];  % Combine past loadings together


    %% Transition equation ------------------------------------------------

    z = F(:,1:r_i);            % Projected data (no lag)
    Z = F(:,r_i+1:r_i*(p+1));  % Data with lag 1

    A_i = zeros(r_i*ppC,r_i*ppC)';  % Initialize transition matrix

    A_temp = inv(Z'*Z)*Z'*z;  % OLS: gives coefficient value AR(p) process
    A_i(1:r_i,1:r_i*p) = A_temp';
    A_i(r_i+1:end,1:r_i*(ppC-1)) = eye(r_i*(ppC-1));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Q_i = zeros(ppC*r_i,ppC*r_i);
    e = z  - Z*A_temp;         % VAR residuals
    Q_i(1:r_i,1:r_i) = cov(e); % VAR covariance matrix

    initV_i = reshape(inv(eye((r_i*ppC)^2)-kron(A_i,A_i))*Q_i(:),r_i*ppC,r_i*ppC);

    % Gives top left block for the transition matrix
    A = blkdiag(A,A_i);
    Q = blkdiag(Q,Q_i);
    V_0 = blkdiag(V_0,initV_i);
end

eyeN = eye(N);  % Used inside observation matrix
eyeN(:,~i_idio) = [];

C=[C eyeN];
C = [C [zeros(nM,5*nQ); kron(eye(nQ),[1 2 3 2 1])]];  % Monthly-quarterly agreggation scheme
R = diag(var(resNaN,'omitnan'));  % Initialize covariance matrix for transition matrix


ii_idio = find(i_idio);    % Indicies for monthly variables
n_idio = length(ii_idio);  % Number of monthly variables
BM = zeros(n_idio);        % Initialize monthly transition matrix values
SM = zeros(n_idio);        % Initialize monthly residual covariance matrix values


for i = 1:n_idio;  % Loop for monthly variables
    % Set observation equation residual covariance matrix diagonal
    R(ii_idio(i),ii_idio(i)) = 1e-04;

    % Subsetting series residuals for series i
    res_i = resNaN(:,ii_idio(i));

    % Returns number of leading/ending zeros
    leadZero = max( find( (1:T)' == cumsum(isnan(res_i)) ) );
    endZero  = max( find( (1:T)' == cumsum(isnan(res_i(end:-1:1))) ) );

    % Truncate leading and ending zeros
    res_i = res(:,ii_idio(i));
    res_i(end-endZero + 1:end) = [];
    res_i(1:leadZero) = [];

    % Linear regression: AR 1 process for monthly series residuals
    BM(i,i) = inv(res_i(1:end-1)'*res_i(1:end-1))*res_i(1:end-1)'*res_i(2:end,:);
    SM(i,i) = cov(res_i(2:end)-res_i(1:end-1)*BM(i,i));  % Residual covariance matrix

end

Rdiag = diag(R);
sig_e = Rdiag(nM+1:N)/19;
Rdiag(nM+1:N) = 1e-04;
R = diag(Rdiag);  % Covariance for obs matrix residuals

% For BQ, SQ
rho0 = 0.1;
temp = zeros(5);
temp(1,1) = 1;

% Blocks for covariance matrices
SQ = kron(diag((1-rho0^2)*sig_e),temp);
BQ = kron(eye(nQ),[[rho0 zeros(1,4)];[eye(4),zeros(4,1)]]);
initViQ = reshape(inv(eye((5*nQ)^2)-kron(BQ,BQ))*SQ(:),5*nQ,5*nQ);
initViM = diag(1./diag(eye(size(BM,1))-BM.^2)).*SM;

% Output
A = blkdiag(A, BM, BQ);                % Observation matrix
Q = blkdiag(Q, SM, SQ);                % Residual covariance matrix (transition)
Z_0 = zeros(size(A,1),1);              % States
V_0 = blkdiag(V_0, initViM, initViQ);  % Covariance of states

end



function [zsmooth, Vsmooth, VVsmooth, loglik] = runKF(Y, A, C, Q, R, Z_0, V_0);
%runKF()    Applies Kalman filter and fixed-interval smoother
%
%  Syntax:
%    [zsmooth, Vsmooth, VVsmooth, loglik] = runKF(Y, A, C, Q, R, Z_0, V_0)
%
%  Description:
%    runKF() applies a Kalman filter and fixed-interval smoother. The
%    script uses the following model:
%           Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
%           Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)

%  Throughout this file:
%    'm' denotes the number of elements in the state vector Z_t.
%    'k' denotes the number of elements (observed variables) in Y_t.
%    'nobs' denotes the number of time periods for which data are observed.
%
%  Input parameters:
%    Y: k-by-nobs matrix of input data
%    A: m-by-m transition matrix 
%    C: k-by-m observation matrix
%    Q: m-by-m covariance matrix for transition equation residuals (mu_t)
%    R: k-by-k covariance for observation matrix residuals (e_t)
%    Z_0: 1-by-m vector, initial value of state
%    V_0: m-by-m matrix, initial value of state covariance matrix
%
%  Output parameters:
%    zsmooth: k-by-(nobs+1) matrix, smoothed factor estimates
%             (i.e. zsmooth(:,t+1) = Z_t|T)
%    Vsmooth: k-by-k-by-(nobs+1) array, smoothed factor covariance matrices
%             (i.e. Vsmooth(:,:,t+1) = Cov(Z_t|T))
%    VVsmooth: k-by-k-by-nobs array, lag 1 factor covariance matrices
%              (i.e. Cov(Z_t,Z_t-1|T))
%    loglik: scalar, log-likelihood
%
%  References:
%  - QuantEcon's "A First Look at the Kalman Filter"
%  - Adapted from replication files for:
%    "Nowcasting", 2010, (by Marta Banbura, Domenico Giannone and Lucrezia 
%    Reichlin), in Michael P. Clements and David F. Hendry, editors, Oxford 
%    Handbook on Economic Forecasting.
%
% The software can be freely used in applications. 
% Users are kindly requested to add acknowledgements to published work and 
% to cite the above reference in any resulting publications

S = SKF(Y, A, C, Q, R, Z_0, V_0);  % Kalman filter
S = FIS(A, S);                     % Fixed interval smoother

% Organize output 
zsmooth = S.ZmT;
Vsmooth = S.VmT;
VVsmooth = S.VmT_1;
loglik = S.loglik;

end

%______________________________________________________________________
function S = SKF(Y, A, C, Q, R, Z_0, V_0)
%SKF    Applies Kalman filter
%
%  Syntax:
%    S = SKF(Y, A, C, Q, R, Z_0, V_0)
%
%  Description:
%    SKF() applies the Kalman filter

%  Input parameters:
%    Y: k-by-nobs matrix of input data
%    A: m-by-m transition matrix 
%    C: k-by-m observation matrix
%    Q: m-by-m covariance matrix for transition equation residuals (mu_t)
%    R: k-by-k covariance for observation matrix residuals (e_t)
%    Z_0: 1-by-m vector, initial value of state
%    V_0: m-by-m matrix, initial value of state covariance matrix
%
%  Output parameters:
%    S.Zm: m-by-nobs matrix, prior/predicted factor state vector
%          (S.Zm(:,t) = Z_t|t-1)
%    S.ZmU: m-by-(nobs+1) matrix, posterior/updated state vector
%           (S.Zm(t+1) = Z_t|t)
%    S.Vm: m-by-m-by-nobs array, prior/predicted covariance of factor
%          state vector (S.Vm(:,:,t) = V_t|t-1)  
%    S.VmU: m-by-m-by-(nobs+1) array, posterior/updated covariance of
%           factor state vector (S.VmU(:,:,t+1) = V_t|t)
%    S.loglik: scalar, value of likelihood function
%    S.k_t: k-by-m Kalman gain
  
%% INITIALIZE OUTPUT VALUES ---------------------------------------------
  % Output structure & dimensions of state space matrix
  [~, m] = size(C);
  
  % Outputs time for data matrix. "number of observations"
  nobs  = size(Y,2);
  
  % Instantiate output
  S.Zm  = nan(m, nobs);       % Z_t | t-1 (prior)
  S.Vm  = nan(m, m, nobs);    % V_t | t-1 (prior)
  S.ZmU = nan(m, nobs+1);     % Z_t | t (posterior/updated)
  S.VmU = nan(m, m, nobs+1);  % V_t | t (posterior/updated)
  S.loglik = 0;

%% SET INITIAL VALUES ----------------------------------------------------
  Zu = Z_0;  % Z_0|0 (In below loop, Zu gives Z_t | t)
  Vu = V_0;  % V_0|0 (In below loop, Vu guvse V_t | t)
  
  % Store initial values
  S.ZmU(:,1)    = Zu;
  S.VmU(:,:,1)  = Vu;

%% KALMAN FILTER PROCEDURE ----------------------------------------------
  for t = 1:nobs
      %%% CALCULATING PRIOR DISTIBUTION----------------------------------
      
      % Use transition eqn to create prior estimate for factor
      % i.e. Z = Z_t|t-1
      Z   = A * Zu;
      
      % Prior covariance matrix of Z (i.e. V = V_t|t-1)
      %   Var(Z) = Var(A*Z + u_t) = Var(A*Z) + Var(\epsilon) = 
      %   A*Vu*A' + Q
      V   = A * Vu* A' + Q; 
      V   =  0.5 * (V+V');  % Trick to make symmetric
      
      %%% CALCULATING POSTERIOR DISTRIBUTION ----------------------------
       
      % Removes missing series: These are removed from Y, C, and R
      [Y_t, C_t, R_t, ~] = MissData(Y(:,t), C, R); 

      % Check if y_t contains no data. If so, replace Zu and Vu with prior.
      if isempty(Y_t)
          Zu = Z;
          Vu = V;
      else  
          % Steps for variance and population regression coefficients:
          % Var(c_t*Z_t + e_t) = c_t Var(A) c_t' + Var(u) = c_t*V *c_t' + R
          VC  = V * C_t';  
          iF  = inv(C_t * VC + R_t);
          
          % Matrix of population regression coefficients (QuantEcon eqn #4)
          VCF = VC*iF;  

          % Gives difference between actual and predicted observation
          % matrix values
          innov  = Y_t - C_t*Z;
          
          % Update estimate of factor values (posterior)
          Zu  = Z  + VCF * innov;
          
          % Update covariance matrix (posterior) for time t
          Vu  = V  - VCF * VC';
          Vu   =  0.5 * (Vu+Vu'); % Approximation trick to make symmetric
          
          % Update log likelihood
          % t
          % llf = -0.5 * size(Y_t, 1) * log(2 * pi) - 0.5*(log(det(iF))  - innov'*iF*innov)
          % fprintf('----------')
          S.loglik = S.loglik + 0.5*(log(det(iF))  - innov'*iF*innov);
      end
      
      %%% STORE OUTPUT----------------------------------------------------
      
      % Store covariance and observation values for t-1 (priors)
      S.Zm(:,t)   = Z;
      S.Vm(:,:,t) = V;

      % Store covariance and state values for t (posteriors)
      % i.e. Zu = Z_t|t   & Vu = V_t|t
      S.ZmU(:,t+1)    = Zu;
      S.VmU(:,:,t+1)  = Vu;
  end 
  
  % Store Kalman gain k_t
  if isempty(Y_t)
      S.k_t = zeros(m,m);
  else
      S.k_t = VCF * C_t;
  end
  
end


%______________________________________________________________________
function S = FIS(A, S)
%FIS()    Applies fixed-interval smoother
%
%  Syntax:
%    S = FIS(A, S)
%
%  Description:
%    SKF() applies a fixed-interval smoother, and is used in conjunction 
%    with SKF(). See  page 154 of 'Forecasting, structural time series models 
%    and the Kalman filter' for more details (Harvey, 1990).
%
%  Input parameters:
%    A: m-by-m transition matrix 
%    S: structure returned by SKF()
%
%  Output parameters:
%    S: FIS() adds the following smoothed estimates to the S structure: 
%    - S.ZmT: m-by-(nobs+1) matrix, smoothed states
%             (S.ZmT(:,t+1) = Z_t|T) 
%    - S.VmT: m-by-m-by-(nobs+1) array, smoothed factor covariance
%             matrices (S.VmT(:,:,t+1) = V_t|T = Cov(Z_t|T))
%    - S.VmT_1: m-by-m-by-nobs array, smoothed lag 1 factor covariance
%               matrices (S.VmT_1(:,:,t) = Cov(Z_t Z_t-1|T))
%
%  Model:
%   Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
%   Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)

%% ORGANIZE INPUT ---------------------------------------------------------

% Initialize output matrices
  [m, nobs] = size(S.Zm);
  S.ZmT = zeros(m,nobs+1);
  S.VmT = zeros(m,m,nobs+1);
  
  % Fill the final period of ZmT, VmT with SKF() posterior values
  S.ZmT(:,nobs+1)   = squeeze(S.ZmU(:, nobs+1));
  S.VmT(:,:,nobs+1) = squeeze(S.VmU(:,:, nobs+1));

  % Initialize VmT_1 lag 1 covariance matrix for final period
  S.VmT_1(:,:,nobs) = (eye(m)-S.k_t) *A*squeeze(S.VmU(:,:,nobs));
  
  % Used for recursion process. See companion file for details
  J_2 = squeeze(S.VmU(:,:,nobs)) * A' * pinv(squeeze(S.Vm(:,:,nobs)));

  %% RUN SMOOTHING ALGORITHM ----------------------------------------------
  
  % Loop through time reverse-chronologically (starting at final period nobs)
    for t = nobs:-1:1
                
        % Store posterior and prior factor covariance values 
        VmU = squeeze(S.VmU(:,:,t));
        Vm1 = squeeze(S.Vm(:,:,t));
        
        % Store previous period smoothed factor covariance and lag-1 covariance
        V_T = squeeze(S.VmT(:,:,t+1));
        V_T1 = squeeze(S.VmT_1(:,:,t));
      
        J_1 = J_2;
                
        % Update smoothed factor estimate
        S.ZmT(:,t) = S.ZmU(:,t) + J_1 * (S.ZmT(:,t+1) - A * S.ZmU(:,t)) ; 
        
        % Update smoothed factor covariance matrix
        S.VmT(:,:,t) = VmU + J_1 * (V_T - Vm1) * J_1';   
      
        if t>1
            % Update weight
            J_2 = squeeze(S.VmU(:, :, t-1)) * A' * pinv(squeeze(S.Vm(:,:,t-1)));
            
            % Update lag 1 factor covariance matrix 
            S.VmT_1(:,:,t-1) = VmU * J_2'+J_1 * (V_T1 - A * VmU) * J_2';
        end
    end

end

    
function [y,C,R,L]  = MissData(y,C,R)
% Syntax:
% Description:
%   Eliminates the rows in y & matrices C, R that correspond to missing 
%   data (NaN) in y
%
% Input:
%   y: Vector of observations at time t
%   C: Observation matrix
%   R: Covariance for observation matrix residuals
%
% Output:
%   y: Vector of observations at time t (reduced)     
%   C: Observation matrix (reduced)     
%   R: Covariance for observation matrix residuals
%   L: Used to restore standard dimensions(n x #) where # is the nr of 
%      available data in y
  
  % Returns 1 for nonmissing series
  ix = ~isnan(y);
  
  % Index for columns with nonmissing variables
  e  = eye(size(y,1));
  L  = e(:,ix);

  % Removes missing series
  y  = y(ix);
  
  % Removes missing series from observation matrix
  C  =  C(ix,:);  
  
  % Removes missing series from transition matrix
  R  =  R(ix,ix);

end

