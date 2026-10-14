% plot_nk3_irf.m - Run the NK model and save its impulse responses to a monetary policy shock
dynare nk3

horizon = 0:15;     % irf=16 -> 16 periods, the first one is the impact
vars    = {'y_gap', 'infl', 'i_nom', 'r_real'};
titles  = {'Output gap (%)', 'Inflation (pp, annualised)', ...
    'Nominal rate (pp, annualised)', 'Real rate (pp, annualised)'};
scale   = [1, 4, 4, 4];     % quarterly rates x 4 = annualised

fig = figure('Position', [100 100 900 650]);
for k = 1:4
    subplot(2, 2, k);
    response = scale(k) * oo_.irfs.([vars{k} '_eps_v']);
    plot(horizon, response, 'LineWidth', 2);
    yline(0, '--k');
    title(titles{k});
    xlabel('Quarters after the shock');
    grid on;
end
sgtitle('NK model: responses to a monetary policy shock (+1 pp annualised)');
exportgraphics(fig, '../figures/nk3_irf_monetary.png', 'Resolution', 300);