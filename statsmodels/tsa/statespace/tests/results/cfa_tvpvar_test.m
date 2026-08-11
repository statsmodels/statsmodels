% % =======================================================================
% % TVP-VAR test results generation
% % 
% % Based on the TVPVAR.m file found at
% % http://joshuachan.org/code/code_TVPVAR.html
% % 
% % See Chan, J.C.C. and Jeliazkov, I. (2009) Efficient Simulation and
% %     Integrated Likelihood Estimation in State Space Models 
% % 
% % =======================================================================

% Code to generate the dataset being used is as follows:
% dta = sm.datasets.macrodata.load_pandas().data
% dta.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
% endog = np.log(dta[['realcons', 'realgdp', 'cpi']]).diff().iloc[1:13] * 400
% endog.to_csv('sp_code/macrodata.csv', index=False, header=False)

clear; clc;
rng(1234)
load macrodata.csv;
Y0 = macrodata(1:3,:);
shortY = macrodata(4:12,:);
[T n] = size(shortY);
Y = reshape(shortY',T*n,1);
p = 1;        % no. of lags
q = n^2*p+n;  % dim of states
Tq = T*q; Tn = T*n; q2 = q^2; nnp1 = n*(n+1);
nloop = 1; burnin = 0;
    %% initialize for storage
store_Omega11 = zeros(nloop-burnin,n,n);
store_Omega22 = zeros(nloop-burnin,q);
store_beta = zeros(nloop-burnin,Tq);
    %% initialize the Markov chain
Omega11 = cov(macrodata);
Omega22 = .01*ones(q,1); % store only the diag elements
invOmega11 = Omega11\speye(n);
invOmega22 = 1./Omega22;
beta = zeros(Tq,1);
    %% prior
nu01 = n+3; S01 = eye(n);
nu02 = 3;  S02 = .005*ones(q,1);
Vbeta = 5*ones(q,1);
    %% construct and compute a few thigns
X = zeros(T,n*p);
for i=1:p
    X(:,(i-1)*n+1:i*n) = [Y0(3-i+1:end,:); shortY(1:T-i,:)];
end
bigG = SURform([ones(n*T,1) kron(X,ones(n,1))]);
H = speye(Tq,Tq) - sparse(q+1:Tq,1:(T-1)*q,ones(1,(T-1)*q),Tq,Tq);
newnu1 = nu01 + T;
newnu2 = nu02 + (T-1)/2;


%% - Iteration 1 ----------------------------------------------------------

% sample beta
invS = sparse(1:Tq,1:Tq,[1./Vbeta; repmat(invOmega22,T-1,1)]'); 
K = H'*invS*H;   
GinvOmega11 = bigG'*kron(speye(T),invOmega11);
GinvOmega11G = GinvOmega11*bigG;
invP = K + GinvOmega11G;
C = chol(invP);                         % so that C'*C = invP
betahat = C\(C'\(GinvOmega11*Y));
state_variates = randn(Tq,1);
beta = betahat + C\state_variates;         % C is upper triangular

% sample Omega11
e1 = reshape(Y-bigG*beta,n,T);
newS1 = S01 + e1*e1';
invOmega11 = wishrnd(newS1\speye(n),newnu1);
Omega11 = invOmega11\speye(n);

% sample Omega22
e2 = reshape(H*beta,q,T);
newS2 = S02 + sum(e2(:,2:end).^2,2)/2;
invOmega22 = gamrnd(newnu2,1./newS2);
Omega22 = 1./invOmega22;

% save results
invP_1 = invP;
posterior_mean_1 = reshape(betahat, q, T);
state_variates_1 = reshape(state_variates, q, T);
beta_1 = reshape(beta, q, T);

S10_1 = newS1;
v10_1 = newnu1;
Omega11_1 = Omega11;

Si0_1 = newS2;
vi0_1 = newnu2;
Omega22_1 = Omega22;


%% - Iteration 2 ----------------------------------------------------------
% sample beta
invS = sparse(1:Tq,1:Tq,[1./Vbeta; repmat(invOmega22,T-1,1)]'); 
K = H'*invS*H;   
GinvOmega11 = bigG'*kron(speye(T),invOmega11);
GinvOmega11G = GinvOmega11*bigG;
invP = K + GinvOmega11G;
aaa = full(invP);
bbb = inv(aaa);
C = chol(invP);                         % so that C'*C = invP
betahat = C\(C'\(GinvOmega11*Y));
state_variates = randn(Tq,1);
beta = betahat + C\state_variates;         % C is upper triangular

% sample Omega11
e1 = reshape(Y-bigG*beta,n,T);
newS1 = S01 + e1*e1';
invOmega11 = wishrnd(newS1\speye(n),newnu1);
Omega11 = invOmega11\speye(n);

% sample Omega22
e2 = reshape(H*beta,q,T);
newS2 = S02 + sum(e2(:,2:end).^2,2)/2;
invOmega22 = gamrnd(newnu2,1./newS2);
Omega22 = 1./invOmega22;

% save results
invP_2 = invP;
posterior_mean_2 = reshape(betahat, q, T);
state_variates_2 = reshape(state_variates, q, T);
beta_2 = reshape(beta, q, T);

S10_2 = newS1;
v10_2 = newnu1;
Omega11_2 = Omega11;

Si0_2 = newS2;
vi0_2 = newnu2;
Omega22_2 = Omega22;

%% - Output ---------------------------------------------------------------
invP = [full(invP_1)' full(invP_2)']';
posterior_mean = [posterior_mean_1' posterior_mean_2']';
state_variates = [state_variates_1' state_variates_2']';
beta = [beta_1' beta_2']';

S10 = [S10_1 S10_2];
v10 = [v10_1 v10_2];
Omega_11 = [Omega11_1 Omega11_2];

Si0 = [Si0_1 Si0_2];
vi0 = [vi0_1 vi0_2];
Omega_22 = [Omega22_1 Omega22_2];

dlmwrite('cfa_tvpvar_invP.csv', invP, 'precision', 15);
dlmwrite('cfa_tvpvar_posterior_mean.csv', posterior_mean, 'precision', 10);
dlmwrite('cfa_tvpvar_state_variates.csv', state_variates, 'precision', 10);
dlmwrite('cfa_tvpvar_beta.csv', beta, 'precision', 10);
dlmwrite('cfa_tvpvar_S10.csv', S10, 'precision', 10);
dlmwrite('cfa_tvpvar_v10.csv', v10, 'precision', 10);
dlmwrite('cfa_tvpvar_Omega_11.csv', Omega_11, 'precision', 10);
dlmwrite('cfa_tvpvar_Si0.csv', Si0, 'precision', 10);
dlmwrite('cfa_tvpvar_vi0.csv', vi0, 'precision', 10);
dlmwrite('cfa_tvpvar_Omega_22.csv', Omega_22, 'precision', 10);
