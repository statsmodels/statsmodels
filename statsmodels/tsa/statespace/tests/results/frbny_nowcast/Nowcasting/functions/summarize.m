function summarize(X,Time,Spec,vintage)
%summarize Display the detail table for data entering the DFM
%
% Description:
%   Display the detail table for the nowcast, decomposing nowcast changes
%   into news and impacts for released data series.

% Print results to command window
fprintf('\n\n\n');
fprintf('Table 2: Data Summary \n');

[T,N] = size(X);
%fprintf('Vintage: %s \n',datestr(datenum(vintage,'yyyy-mm-dd'),'mmmm dd, yyyy'));
fprintf('N = %4d data series \n',N);
fprintf('T = %4d observations from %10s to %10s \n',T,datestr(Time(1),'yyyy-mm-dd'),datestr(Time(end),'yyyy-mm-dd'));


fprintf('%30s | %17s    %12s    %10s    %8s    %8s    %8s    %8s \n',...
        'Data Series','Observations','Units','Frequency','Mean','Std. Dev.','Min','Max');
fprintf([repmat('-',1,130) '\n']);

for i = 1:N
    
    % time indexes for which there are observed values for series i
    t_obs = ~isnan(X(:,i));
    
    data_series = Spec.SeriesName{i};
    if length(data_series) > 30
        data_series = [data_series(1:27) '...'];
    end
    series_id   = Spec.SeriesID{i};
    if length(series_id) > 28
        series_id = [series_id(1:25) '...'];
    end
    series_id   = ['[' series_id ']'];
    num_obs     = length(X(t_obs,i));
    freq        = Spec.Frequency{i};
    if strcmp('m',freq)
        format_date = 'mmm yyyy';
        frequency   = 'Monthly';
    elseif strcmp('q',freq)
        format_date = 'QQ yyyy';
        frequency   = 'Quarterly';
    end
    units       = Spec.Units{i};
    transform   = Spec.Transformation{i};
    % display transformed units
    if strfind(units,'Index')
        units_transformed = 'Index';
    elseif strcmp('chg',transform)
        if strfind(units,'%')
            units_transformed = 'Ppt. change';
        else
            units_transformed = 'Level change';
        end
    elseif strcmp('pch',transform) && strcmp('m',freq)
        units_transformed = 'MoM%';
    elseif strcmp('pca',transform) && strcmp('q',freq)
        units_transformed = 'QoQ% AR';
    else
        if length([units ' (' transform ')']) > 12
            units_transformed = [units(1:6) ' (' transform ')'];
        else
            units_transformed = [units ' (' transform ')'];
        end
    end
    t_obs_start = find(t_obs,1,'first');
    t_obs_end   = find(t_obs,1,'last');
    obs_date_start  = datestr(Time(t_obs_start),format_date);
    obs_date_end    = datestr(Time(t_obs_end  ),format_date);
    date_range  = [obs_date_start '-' obs_date_end];
    y           = X(t_obs,i);
    d           = Time(t_obs);
    mean_series = mean(y,'omitnan');
    stdv_series = std(y,'omitnan');
    [min_series,t_min] = min(y);
    [max_series,t_max] = max(y);
    min_date = datestr(d(t_min),format_date);
    max_date = datestr(d(t_max),format_date);
    fprintf('%30s | %17d    %12s    %10s    %8.1f    %8.1f    %8.1f    %8.1f \n',...
            data_series,num_obs,units_transformed,frequency,mean_series,stdv_series,min_series,max_series);
    fprintf('%30s | %17s    %12s    %10s    %8s    %8s    %8s    %8s \n',...
            series_id,date_range,'','','','',min_date,max_date);
    
end

fprintf('\n\n\n');

end

