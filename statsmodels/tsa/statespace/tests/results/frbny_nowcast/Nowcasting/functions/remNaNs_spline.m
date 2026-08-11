%%%  Replication files for:
%%%  ""Nowcasting", 2010, (by Marta Banbura, Domenico Giannone and Lucrezia Reichlin), 
%%% in Michael P. Clements and David F. Hendry, editors, Oxford Handbook on Economic Forecasting.
%%%
%%% The software can be freely used in applications. 
%%% Users are kindly requested to add acknowledgements to published work and 
%%% to cite the above reference in any resulting publications
%
%Description:
%
%remNaNs    Treats NaNs in dataset for use in DFM.
%
%  Syntax:
%    [X,indNaN] = remNaNs(X,options)
%
%  Description:
%    remNaNs() processes NaNs in a data matrix X according to 5 cases (see
%    below for details). These are useful for running functions in the 
%    'DFM.m' file that do not take missing value inputs.
%
%  Input parameters:
%    X (T x n): Input data where T gives time and n gives the series. 
%    options: A structure with two elements:
%      options.method (numeric):
%      - 1: Replaces all missing values using filter().
%      - 2: Replaces missing values after removing trailing and leading
%           zeros (a row is 'missing' if >80% is NaN)
%      - 3: Only removes rows with leading and closing zeros
%      - 4: Replaces missing values after removing trailing and leading
%           zeros (a row is 'missing' if all are NaN)
%      - 5: Replaces missing values with spline() then runs filter().
%
%      options.k (numeric): remNaNs() relies on MATLAB's filter function
%      for the 1-D filter. k controls the rational transfer function
%      argument's numerator (the denominator is set to 1). More
%      specifically, the numerator takes the form 'ones(2*k+1,1)/(2*k+1)'
%      For additional help, see MATLAB's documentation for filter().
%
%  Output parameters:
%    X: Outputted data. 
%    indNaN: A matrix indicating the location for missing values (1 for NaN).  

function [X,indNaN] = remNaNs_spline(X,options)
[T,N]=size(X);    % Gives dimensions for data input
k=options.k;      % Inputted options
indNaN=isnan(X);  % Returns location of NaNs

switch options.method
    case 1  % replace all the missing values
        for i = 1:N  % loop through columns
            x = X(:,i); 
            x(indNaN(:,i))= nan_median(x);  % Replace missing values series median 
            x_MA =filter (ones(2*k+1,1)/(2*k+1), 1, [x(1)*ones(k,1);x;x(end)*ones(k,1)]);
            x_MA=x_MA(2*k+1:end);  % Match dimensions
            % Replace missing observations with filtered values
            x(indNaN(:,i))=x_MA(indNaN(:,i));
            X(:,i)=x;  % Replace vector
        end
    case 2 %replace missing values after removing leading and closing zeros
        
        % Returns row sum for NaN values. Marks true for rows with more
        % than 80% NaN
        rem1=(sum(indNaN,2)>N*0.8);
        nanLead =(cumsum(rem1)==(1:T)');
        nanEnd =(cumsum(rem1(end:-1:1))==(1:T)');
        nanEnd = nanEnd(end:-1:1);  % Reverses nanEnd
        nanLE = (nanLead | nanEnd);
        
        % Subsets X for for 
        X(nanLE,:) = [];
        indNaN = isnan(X);  % Index for missing values
        % Loop for each series
        for i = 1:N  
            x = X(:,i);
            isnanx = isnan(x);
            t1 = min(find(~isnanx));  % First non-NaN entry 
            t2 = max(find(~isnanx));  % Last non-NaN entry
            % Interpolates without NaN entries in beginning and end
            x(t1:t2) = spline(find(~isnanx),x(~isnanx),(t1:t2)');
            isnanx = isnan(x);
            % replace NaN observations with median
            x(isnanx) = median(x,'omitnan');
            % Apply filter
            x_MA = filter (ones(2*k+1,1)/(2*k+1),1,[x(1)*ones(k,1);x;x(end)*ones(k,1)]);
            x_MA = x_MA(2*k+1:end);
            % Replace nanx with filtered observations
            x(isnanx) = x_MA(isnanx);
            X(:,i) = x;
        end
    case 3 %only remove rows with leading and closing zeros
        rem1=(sum(indNaN,2)==N);
        nanLead=(cumsum(rem1)==(1:T)');
        nanEnd=(cumsum(rem1(end:-1:1))==(1:T)');
        nanEnd=nanEnd(end:-1:1);
        nanLE=(nanLead | nanEnd);
        X(nanLE,:)=[];
        indNaN = isnan(X);
    case 4  %remove rows with leading and closing zeros & replace missing values
        rem1=(sum(indNaN,2)==N);
        nanLead=(cumsum(rem1)==(1:T)');
        nanEnd=(cumsum(rem1(end:-1:1))==(1:T)');
        nanEnd=nanEnd(end:-1:1);
        nanLE=(nanLead | nanEnd);
        X(nanLE,:)=[];
        indNaN=isnan(X);
        for i = 1:N  
            x = X(:,i);
            isnanx = isnan(x);
            t1 = min(find(~isnanx));
            t2 = max(find(~isnanx));
            x(t1:t2) = spline(find(~isnanx),x(~isnanx),(t1:t2)');
            isnanx = isnan(x);
            x(isnanx)=nan_median(x);
            x_MA =filter (ones(2*k+1,1)/(2*k+1),1,[x(1)*ones(k,1);x;x(end)*ones(k,1)]);
            x_MA=x_MA(2*k+1:end);
            x(isnanx)=x_MA(isnanx);
            X(:,i)=x;
        end
    case 5 %replace missing values  
        indNaN=isnan(X);
        for i = 1:N  
            x = X(:,i);
            isnanx = isnan(x);
            t1 = min(find(~isnanx));
            t2 = max(find(~isnanx));
            x(t1:t2) = spline(find(~isnanx),x(~isnanx),(t1:t2)');
            isnanx = isnan(x);
            x(isnanx)=nan_median(x);
            x_MA =filter (ones(2*k+1,1)/(2*k+1),1,[x(1)*ones(k,1);x;x(end)*ones(k,1)]);
            x_MA=x_MA(2*k+1:end);
            x(isnanx)=x_MA(isnanx);
            X(:,i)=x;
        end
end
