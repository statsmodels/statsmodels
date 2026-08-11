function [X,Time,Z] = load_data(datafile,Spec,sample,loadExcel)
%loadData Load vintage of data from file and format as structure
%
% Description:
%
%   Load data from file
%
% Input Arguments:
%
%   datafile - filename of Microsoft Excel workbook file
%
% Output Arguments:
%
% Data - structure with the following fields:
%   .    X : T x N numeric array, transformed dataset
%   . Time : T x 1 numeric array, date number with observation dates
%   .    Z : T x N numeric array, raw (untransformed) dataset

fprintf('Loading data... \n');

if nargin < 4
    loadExcel = 0;
end

ext = datafile(find(datafile=='.',1,'last'):end); % file extension

idx = strfind(datafile,filesep); idx = idx(2);
datafile_mat = fullfile(datafile(1:idx),'mat',[strtok(datafile(idx+1:end),'.') '.mat']);
if exist(datafile_mat,'file')==2 && ~loadExcel
    % Load raw data from MATLAB formatted binary (.mat) file
    d = load(datafile_mat);
    Z    = d.Z;
    Time = d.Time;
    Mnem = d.Mnem;
elseif any(strcmpi(ext,{'.xlsx','.xls'}))
    % Read raw data from Excel file
    [Z,Time,Mnem] = readData(datafile);
    save(datafile_mat,'Z','Time','Mnem');
    %    Z : raw (untransformed) observed data
    % Time : observation periods for the time series data
    % Mnem : series ID for each variable
else
    error('Only Microsoft Excel workbook files supported.');
end

% Sort data based on model specification
Z = sortData(Z,Mnem,Spec);
clear Mnem; % since now Mnem == Spec.SeriesID

% Transform data based on model specification
[X,Time,Z] = transformData(Z,Time,Spec);

% Drop data not in estimation sample
if nargin == 3
    [X,Time,Z] = dropData(X,Time,Z,sample);
end

end


function [Z,Time,Mnem] = readData(datafile)
%readData Read data from Microsoft Excel workbook file

[DATA,TEXT] = xlsread(datafile,'data');
Mnem = TEXT(1,2:end);
if ispc
    Time = datenum(TEXT(2:end,1),'mm/dd/yyyy');
    Z    = DATA;
else
    Time = DATA(:,1) + datenum(1899,12,31) - 1;
    Z    = DATA(:,2:end);
end

end

function [Z,Mnem] = sortData(Z,Mnem,Spec)
%sortData Sort series by order of model specification

% Drop series not in Spec
inSpec = ismember(Mnem,Spec.SeriesID);
Mnem(~inSpec) = [];
Z(:,~inSpec)  = [];

% Sort series by ordering of Spec
N = length(Spec.SeriesID);
permutation = NaN(1,N);
for i = 1:N
    idxSeriesSpec = find(strcmp(Spec.SeriesID{i},Mnem));
    permutation(i) = idxSeriesSpec;
end

Mnem = Mnem(permutation);
Z    = Z(:,permutation);


end


function [X,Time,Z] = transformData(Z,Time,Spec)
%transformData Transforms each data series based on Spec.Transformation
%
% Input Arguments:
%
%      Z : T x N numeric array, raw (untransformed) observed data
%   Spec : structure          , model specification
%
% Output Arguments:
%
%      X : T x N numeric array, transformed data (stationary to enter DFM)

T = size(Z,1);
N = size(Z,2);

X = NaN(T,N);
for i = 1:N
    formula = Spec.Transformation{i};
    freq    = Spec.Frequency{i};
    switch freq  % time step for different frequencies based on monthly time
        case 'm' % Monthly
            step = 1;
        case 'q' % Quarterly
            step = 3;
    end
    t1 = step;    % assume monthly observations start at beginning of quarter
    n  = step/12; % number of years, needed to compute annual % changes
    series  = Spec.SeriesName{i};
    switch formula
        case 'lin' % Levels (No Transformation)
            X(:,i) = Z(:,i);
        case 'chg' % Change (Difference)
            X(t1:step:T,i) = [NaN; Z(t1+step:step:T,i) - Z(t1:step:T-t1,i)];
        case 'ch1' % Year over Year Change (Difference)
            X(12+t1:step:T,i) = Z(12+t1:step:T,i) - Z(t1:step:T-12,i);
        case 'pch' % Percent Change
            X(t1:step:T,i) = 100*[NaN; Z(t1+step:step:T,i) ./ Z(t1:step:T-t1,i) - 1];
        case 'pc1' % Year over Year Percent Change
            X(12+t1:step:T,i) = 100*(Z(12+t1:step:T,i) ./ Z(t1:step:T-12,i) - 1);
        case 'pca' % Percent Change (Annual Rate)
            X(t1:step:T,i) = 100*[NaN; (Z(t1+step:step:T,i) ./ Z(t1:step:T-step,i)).^(1/n) - 1];
        case 'log' % Natural Log
            X(:,i) = log(Z(:,i));
        otherwise
            warning(['Transformation ''' formula ''' not found for ' series '. ' 'Using untransformed data.']);
            X(:,i) = Z(:,i);
    end
end


% Drop first quarter of observations
% since transformations cause missing values
Time = Time(4:end);
Z    = Z(4:end,:);
X    = X(4:end,:);

end


function [X,Time,Z] = dropData(X,Time,Z,sample)
%dropData Remove data not in estimation sample

idxDrop = Time < sample;

Time(idxDrop) = [];
X(idxDrop,:)  = [];
Z(idxDrop,:)  = [];

end

