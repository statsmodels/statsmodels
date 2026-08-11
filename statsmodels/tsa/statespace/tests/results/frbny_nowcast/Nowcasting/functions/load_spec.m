function Spec = load_spec(specfile)
%loadSpec Load model specification for a dynamic factor model (DFM)
%
% Description:
%
%   Load model specification  'Spec' from a Microsoft Excel workbook file
%   given by 'filename'.
%
% Input Arguments:
%
%   filename -
%
% Output Arguments:
%
% spec - 1 x 1 structure with the following fields:
%     . series_id
%     . name
%     . frequency
%     . units
%     . transformation
%     . category
%     . blocks
%     . BlockNames
%

Spec = struct;
[~,~,raw] = xlsread(specfile,'','','basic');
header    = strrep(raw(1,:),' ','');
raw       = raw(2:end,:);

% Find and drop series from Spec that are not in Model
Model = cell2mat(raw(:,strcmpi('Model',header)));
raw(Model==0,:) = [];

% Parse fields given by column names in Excel worksheet
fldnms = {'SeriesID','SeriesName','Frequency','Units','Transformation','Category'};
for iField = 1:numel(fldnms)
    fld = fldnms{iField};
    jCol = find(strcmpi(fld,header),1);
    if isempty(jCol)
        error([fld ' column missing from model specification.']);
        % I guess Title and Units are not necessary though
    else
        Spec.(fld) = raw(:,jCol);
    end
end

% Parse blocks.
jColBlock = strncmpi('Block',header,length('Block'));
Blocks = cell2mat(raw(:,jColBlock));
Blocks(isnan(Blocks)) = 0;
if ~all(Blocks(:,1)==1)
    error('All variables must load on global block.');
else
    Spec.Blocks = Blocks;
end

% Sort all fields of 'Spec' in order of decreasing frequency
frequency = {'d','w','m','q','sa','a'};
permutation = [];
for iFreq = 1:numel(frequency)
    permutation = [permutation; find(strcmp(frequency{iFreq},Spec.Frequency))]; %#ok<AGROW>
end

fldnms = fieldnames(Spec);
for iField = 1:numel(fldnms)
    fld = fldnms{iField};
    Spec.(fld) = Spec.(fld)(permutation,:);
end

Spec.BlockNames = regexprep(header(jColBlock),'Block\d-','');

Spec.UnitsTransformed = Spec.Transformation;
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'lin','Levels (No Transformation)');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'chg','Change (Difference)');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'ch1','Year over Year Change (Difference)');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'pch','Percent Change');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'pc1','Year over Year Percent Change');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'pca','Percent Change (Annual Rate)');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'cch','Continuously Compounded Rate of Change');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'cca','Continuously Compounded Annual Rate of Change');
Spec.UnitsTransformed = strrep(Spec.UnitsTransformed,'log','Natural Log');

% Summarize model specification
fprintf('Table 1: Model specification \n');
try
    tabular = table(Spec.SeriesID,Spec.SeriesName,Spec.Units,Spec.UnitsTransformed,...
                'VariableNames',{'SeriesID','SeriesName','Units','Transformation'});
    disp(tabular);
catch
end

end

