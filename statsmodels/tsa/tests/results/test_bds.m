%
% BDS test for IID time series
%
% Calculates and outputs BDS test statistics and p-values for several
% datasets.
%
% Requires bds function, from Kanzler.
%
% References
% ----------
%
% Kanzler, Ludwig. 1998.
% BDS: MATLAB Module to Calculate Brock, Dechert & Scheinkman Test for
% Independence.
% Statistical Software Components. Boston College Department of Economics.
% http://ideas.repec.org/c/boc/bocode/t871803.html.
%
%

in = csvread('bds_data.csv');

sequence = in(1:25,1);
normal   = in(1:25,2);
combined = in(1:50,3);
gdpc1    = in(1:end,4);

[w_s, sig_s, c_s, c1_s, k_s] = bds(sequence, 5);
[w_n, sig_n, c_n, c1_n, k_n] = bds(normal, 5);
[w_c, sig_c, c_c, c1_c, k_c] = bds(combined, 5);
[w_g, sig_g, c_g, c1_g, k_g] = bds(gdpc1, 5);

out_s = [repmat(1,1,5); 1:5; ...
         NaN w_s; NaN sig_s; NaN c_s; c1_s; repmat(k_s,1,5)]';
out_n = [repmat(2,1,5); 1:5; ...
         NaN w_n; NaN sig_n; NaN c_n; c1_n; repmat(k_n,1,5)]';
out_c = [repmat(3,1,5); 1:5; ...
         NaN w_c; NaN sig_c; NaN c_c; c1_c; repmat(k_c,1,5)]';
out_g = [repmat(4,1,5); 1:5; ...
         NaN w_g; NaN sig_g; NaN c_g; c1_g; repmat(k_g,1,5)]';

out = [out_s; out_n; out_c; out_g];

dlmwrite('bds_results.csv', out, 'precision', '%.8f');
