close all;

create_set_of_plots('Unif_Small_p', ...
    {'Coxian20_1', 'General20_1'}, ...
    {'GC(20)', 'GeneralPH(20)'});

create_set_of_plots('Unif_Medium_p', ...
    {'Coxian35_1', 'Coxian40_1'}, ...
    {'GC(35)', 'GC(40)'});

create_set_of_plots('Unif_Large_p', ...
    {'Coxian50_1', 'Coxian75_1', 'Coxian100_1'}, ...
    {'GC(50)', 'GC(75)', 'GC(100)'});

function create_set_of_plots(plotName, fitNames, fitFullNames)
    [~, ~, lateages] = get_ages();
    [~, ~, latedeaths] = get_deaths();
    offset = min(lateages);
    
    legendEntries = {'Target'}; 
    legendEntries = [legendEntries, fitFullNames];

    startVecs = cell(1, length(fitNames));
    Ts = cell(1, length(fitNames));
    
    % Load in all the fits.
    for i = 1:length(fitNames)
        testName = fitNames{i};
        fprintf(1, 'Loading %s\n', testName);

        fitName = strcat(testName, '_fit.csv');
        if ~exist(fitName, 'file')
            disp('Test was not run..');
            continue
        end

        phases = load(fitName);
        startVecs{i} = phases(:,1);
        Ts{i} = phases(:,2:end);
        size(startVecs{i})
    end
    
    % Calculate the empirical pdf, cdf, hazard rates.
    dt = 1/100;
    xs = linspace(dt, length(latedeaths), length(latedeaths)*100);
    pdfs = latedeaths(ceil(xs)); pdfs = pdfs / (sum(pdfs)*dt);
    fprintf(1, 'sum of jagged one is %f\n', sum(pdfs)*dt);
    cdfs = cumsum(pdfs)*dt;
    hazards = pdfs ./ (1 - cdfs);
    
    % Remove jaggedness from hazard rates.
    sparseInds = 1:ceil(1/dt):length(xs);
    xsSparse = xs(sparseInds);
    hazardsSparse = hazards(sparseInds);
    
    % Plot the probability density functions.
    figure(); clf; hold on;
    
    pos=get(gca, 'Position'); 
    pos(end)=pos(end)*0.9;
    pos(end-1)=pos(end-1)*0.5; 
    set(gca, 'Position', pos);

    handles = zeros(length(legendEntries), 1);
    handles(1) = plot(offset + xs, pdfs, 'LineWidth', 2);

    for i = 1:length(Ts)
        handles(i+1) = plot_phase_type_pdf(Ts{i}, startVecs{i}, offset);
    end
    
%     legend(legendEntries, 'Location','NorthWest');
    axis([offset offset+76 0 0.040]);

    save_plot_as_pdf(strcat(plotName, '_pdf.pdf'))
    
    % Plot the cumulative density functions.
    figure(); clf; hold on;
    
    pos=get(gca, 'Position'); 
    pos(end)=pos(end)*0.55;
    pos(end-1)=pos(end-1)*0.55; 
    set(gca, 'Position', pos);
    
    handles = zeros(length(legendEntries), 1);
    handles(1) = plot(offset + xs, cdfs, 'LineWidth', 2);
    
    for i = 1:length(Ts)
        handles(i+1) = plot_phase_type_cdf(Ts{i}, startVecs{i}, offset);
    end
%     legend(legendEntries, 'Location','NorthWest');
    axis([offset offset+76 0 1]);
    save_plot_as_pdf(strcat(plotName, '_cdf.pdf'));

    % Plot the hazard rate functions.
    figure(); clf; hold on;

    pos=get(gca, 'Position'); 
    pos(end)=pos(end)*0.9;
    pos(end-1)=pos(end-1)*0.5; 
    set(gca, 'Position', pos);

    handles = zeros(length(legendEntries), 1);
    handles(1) = plot(offset + xsSparse, hazardsSparse, 'LineWidth', 2);

    for i = 1:length(Ts)
        handles(i+1) = plot_phase_type_hazard(Ts{i}, startVecs{i}, offset);
    end
%     legend(legendEntries, 'Location','EastOutside');
    axis([offset offset+76 0 0.5]);
    
    save_plot_as_pdf(strcat(plotName, '_hazard.pdf'));
    
    figure; hold on; 
    for i = 1:length(legendEntries)
        h(i) = plot(NaN,NaN, 'LineWidth', 2);
    end
    ax = gca; ax.Visible = 'off';
    legend(h, legendEntries, 'Orientation','horizontal');
    legend boxoff 
    save_plot_as_pdf(strcat(plotName, '_legend.pdf'));
end
    
function [allages, earlyages, lateages] = get_ages()
    allages = 0:110;
    earlyages = 0:34;
    lateages = 35:110;
end

function [alldeaths, earlydeaths, latedeaths] = get_deaths()
    earlydeaths = [204217,13157,11971,10981,10171,9525,9028,8664,8420,8278,8225,8247,8328,8452,8606,8776,8962,9165,9391,9635,9906,10202,10525,10882,11271,11698,12166,12678,13240,13854,14526,15264,16069,16950,17915];
    latedeaths = [18969,20122,21381,22758,24261,25902,27692,29647,31776,34097,36625,39377,42370,45623,49155,52989,57143,61642,66506,71760,77427,83526,90082,97114,104638,112672,121223,130300,139900,150015,160626,171704,183202,195065,207212,219546,231946,244269,256342,267971,278929,288970,297821,305198,310798,314327,315496,314046,309761,302489,292155,278791,262541,243675,222592,199816,175968,151749,127886,105091,84004,65145,48867,35348,24568,16344,10366,6238,3543,1890,941,435,184,72,25,11];
    alldeaths = [earlydeaths, latedeaths];
end


function h = plot_phase_type_pdf(T, a, shiftby, scaleby)
    p = size(T, 1);
    if nargin < 2
        a = ((1:p)==1)';
    end
    if nargin < 3
        shiftby = 0;
    end
    if nargin < 4
        scaleby = 1;
    end
    
    n = 1500;
    
    t = -T*ones(size(a));
    vv = -a'*inv(T)*ones(size(a));
    var = 2*a'*inv(T*T)*ones(size(a))-vv*vv;
    std=sqrt(var);
    truncpoint = vv+4*sqrt(var);
    dt = truncpoint/n;
    
    for i=1:n+1 
        y(i)=dt*(i-1);
        f(i)=a'*expm(T*y(i))*t;
    end
    disp('Total pdf')
    disp(sum(f * scaleby * dt))
    h = plot(y + shiftby,f * scaleby, 'LineWidth', 2);
end


function h = plot_phase_type_cdf(T, a, shiftby, scaleby)
    p = size(T, 1);
    if nargin < 2
        a = ((1:p)==1)';
    end
    if nargin < 3
        shiftby = 0;
    end
    if nargin < 4
        scaleby = 1;
    end
    
    n = 1500;
    
    t = -T*ones(size(a));
    vv = -a'*inv(T)*ones(size(a));
    var = 2*a'*inv(T*T)*ones(size(a))-vv*vv;
    std=sqrt(var);
    truncpoint = vv+4*sqrt(var);
    dt = truncpoint/n;
    
    for i=1:n+1 
        y(i)=dt*(i-1);
        f(i) = 1 - a' * expm(T*y(i))*ones(size(a));
    end

    h = plot(y + shiftby,f * scaleby, 'LineWidth', 2);
end


function h = plot_phase_type_hazard(T, a, shiftby, scaleby)
    p = size(T, 1);
    if nargin < 2
        a = ((1:p)==1)';
    end
    if nargin < 3
        shiftby = 0;
    end
    if nargin < 4
        scaleby = 1;
    end
    
    n = 1500;
    
    t = -T*ones(size(a));
    vv = -a'*inv(T)*ones(size(a));
    var = 2*a'*inv(T*T)*ones(size(a))-vv*vv;
    std=sqrt(var);
    truncpoint = vv+4*sqrt(var);
    dt = truncpoint/n;
    
    for i=1:n+1 
        y(i)=dt*(i-1);
        f(i) = (a'*expm(T*y(i))*t) / (a'*expm(T*y(i))*ones(size(a)));
    end

    h = plot(y + shiftby,f * scaleby, 'LineWidth', 2);
end

function save_plot_as_pdf(name)
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    print(name, '-dpdf', '-bestfit')
end

