clear
close all

%% load
data = load('LML_eta.mat');

LML = data.LML;
lens = data.len_s;
noise = data.noise;

[thetagrid_XX, thetagrid_YY] = meshgrid(lens,noise);

LML_max = max(LML,[],'all')
LML_min = min(LML,[],'all')

% levels = logspace(0,5,50);
levels = linspace(-5e2,5e2,100);

figure;
contour(thetagrid_XX, thetagrid_YY, LML, levels, 'LineWidth', 2)

% optimized theta
% opt_theta = [7.74,1.94e-07]; %eta
% opt_theta = [0.922,0.000116]; %kappa
% opt_theta = [0.417,0.0118]; %lnZ
opt_theta = [0.373,0.0183]; %lnA
hold on
plot(opt_theta(1),opt_theta(2),'rx','LineWidth',2,'MarkerSize',20)

%% figure settings
% load tw_flag.mat
cm_hot = hot(256);
cm_cold = cm_hot(:,[3 2 1]);
%cm_rainbow = readmatrix('cm_rainbow');
%cm_rainbow_rgb = cm_rainbow(1:256,1:3);
colormap(parula)
% set(gca,'ColorScale','log')
c = colorbar;
c.LineWidth = 2;
c.Title.String = "\rho";
% caxis([-1e5 1e4])
colorbar off

box on
xlim([1e-2 1e1])
ylim([1e-3 1e-1])
xticks(10.^[-20:20])
yticks(10.^[-20:20])
% xlabel('True value (\sigma)','FontSize',24)
% ylabel('Inferred value (\sigma)','FontSize',24)
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
% set(gca,'ColorScale','log')

ax1 = gca;
ax1.Position = [0.16 0.16 0.7 0.7];

set(gcf,'Position',[2000,100,600,600])
set(gca,'FontSize',32,'FontName','Times New Roman')
set(gca,'LineWidth',2)