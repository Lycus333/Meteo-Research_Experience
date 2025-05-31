
%% 二级回归：大样本（选1）与小样本（选2）分支处理
clc;
clear;
load('25.5.30.mat');

% 如果需要，加载其他相关数据
% load('Trenddata.mat');
load('F107data.mat');
% load('QBO30data.mat');
% load('QBO10data.mat');
% load('ENSOdata.mat');

% 提示用户选择时间段
section = input('请选择时间段（输入1:2-102；输入2:103-120）: ');

if section == 1
    % ====== 分支1：大样本模式 ======
    monthbegin = 2;
    monthend   = 102;
    small_sample_flag = false;

    % 计算样本长度
    xend = monthend - monthbegin + 1;

    % 计算全局均值（用于后续百分比计算）
    sectionmean = mean( ...
        mean( ...
            squeeze(mean(all_point_f(:,:,monthbegin:monthend), 1, 'omitnan')), ...
        'omitnan'), ...
    'omitnan');

    % 计算目标变量Y：每月平均后的值
    meandh = squeeze(mean(mean(all_point_f(:,:,monthbegin:monthend), 1, 'omitnan'), 2, 'omitnan'))';
    meandh(meandh == 0) = NaN;
    Y = meandh;           % 1×xend 向量
    Y = Y(:);             % 转为 xend×1

    % 构造时间变量
    time = (1:xend)';     % xend×1

    % 构造F107指数：取奇数索引，去均值
    F107data_n = F107data - mean(F107data);
    F107data_f = NaN(120, 1);
    for n = 1:120
        F107data_f(n) = F107data_n(2*n - 1);
    end
    F107_sel = F107data_f(monthbegin:monthend);

    % 构造季节项
    seasonality6a  = cos(2*pi*2*time/6);
    seasonality6b  = sin(2*pi*2*time/6);
    seasonality12a = cos(2*pi*2*time/12);
    seasonality12b = sin(2*pi*2*time/12);

    % 第一阶段：不含时间项的多元回归 （X_no_time）
    X_no_time = [ ...
        ones(xend,1), ...
        seasonality6a, seasonality6b, ...
        seasonality12a, seasonality12b, ...
        F107_sel ...
    ];  % xend×6

    [b_no_time, ~, residuals_no_time, ~, stats_no_time] = regress(Y, X_no_time);

    % 第二阶段：对残差做线性回归，提取时间趋势
    X_time = [ones(xend,1), time];  % xend×2
    [b_time, bint_time, ~, ~, stats_time] = regress(residuals_no_time, X_time);

    % 残差正态性检验
    final_residuals = residuals_no_time - X_time * b_time;

    % Jarque-Bera 检验（需 Econometrics 工具箱）
    [~, jb_pvalue] = jbtest(final_residuals);

    % 绘制残差诊断图
    figure('Name','残差正态性诊断');
    subplot(1,3,1);
    qqplot(final_residuals);
    title(['Q-Q图 (JB p=',num2str(jb_pvalue,3),')']);
    grid on;

    subplot(1,3,2);
    histfit(final_residuals, 10);
    title('残差分布直方图');
    xlabel('残差值');
    ylabel('频数');

    subplot(1,3,3);
    pd = fitdist(final_residuals,'Kernel','Kernel','normal');
    x_values = linspace(min(final_residuals), max(final_residuals), 100);
    pdf_values = pdf(pd, x_values);
    plot(x_values, pdf_values, 'LineWidth', 2);
    hold on;
    plot(x_values, normpdf(x_values, mean(final_residuals), std(final_residuals)), 'r--');
    title('核密度估计 vs 正态分布');
    legend('残差密度','理论正态','Location','best');

    % 计算每十年增长率与百分比
    decade_growth = b_time(2) * 60;  % 线性项系数 × 60（月→十年）
    percent_growth = decade_growth / sectionmean * 100;

    % 输出统计结果
    disp('===== 大样本模式：二级回归结果 =====');
    fprintf('Jarque-Bera检验 p-value: %.4f   ', jb_pvalue);
    if jb_pvalue > 0.05
        disp('残差近似正态 (p > 0.05)');
    else
        disp('残差偏离正态 (p ≤ 0.05)');
    end
    fprintf('偏度: %.2f   峰度: %.2f\n', skewness(final_residuals), kurtosis(final_residuals));

    disp(['二级回归每十年增长量: ', num2str(decade_growth)]);
    disp(['二级回归每十年增长百分比: ', num2str(percent_growth), '%']);

    % 拟合优度计算
    Y_no_time_fit = X_no_time * b_no_time;            % 第一阶段拟合值
    Y_time_fit    = X_time * b_time;                  % 第二阶段时间趋势拟合
    Y_total_fit   = Y_no_time_fit + Y_time_fit;       % 合并后的完整拟合

    R2_no_time    = 1 - sum((Y - Y_no_time_fit).^2) / sum((Y - mean(Y)).^2);
    R2_total      = 1 - sum((Y - Y_total_fit).^2) / sum((Y - mean(Y)).^2);
    R2_time_contr = R2_total - R2_no_time;

    disp(['不含时间项模型的 R²: ',     num2str(R2_no_time)]);
    disp(['完整模型的 R²: ',          num2str(R2_total)]);
    disp(['时间项贡献的 R² 增量: ',   num2str(R2_time_contr)]);

    % 可视化拟合结果
    figure('Name','拟合效果对比');
    subplot(2,1,1);
    plot(Y,           'b-', 'LineWidth', 1.5, 'DisplayName', '原始数据'); hold on;
    plot(Y_total_fit, 'r-', 'LineWidth', 1.5, 'DisplayName', '完整拟合'); 
    plot(Y_no_time_fit,'g--','LineWidth', 1,   'DisplayName', '无时间项拟合');
    legend('show');
    title('数据拟合效果对比');
    xlabel('时间索引'); ylabel('变量值');

    subplot(2,1,2);
    plot(residuals_no_time, 'b-', 'LineWidth', 1.5, 'DisplayName', '去除已知项后残差'); hold on;
    plot(Y_time_fit,        'r-', 'LineWidth', 1.5, 'DisplayName', '时间线性拟合');
    legend('show');
    title('残差的线性时间趋势');
    xlabel('时间索引'); ylabel('残差值');

    % F107 与时间的相关性
    F107_time_corr = corrcoef(F107_sel, time);
    disp(['F107 与时间的相关系数: ', num2str(F107_time_corr(1,2))]);
    disp(' ');

    % 对比不同模型结果
    disp('=========== 不同方法对比 ===========');
    % 1. 完整多元回归模型（含时间项）
    X_full = [ones(xend,1), time, seasonality6a, seasonality6b, seasonality12a, seasonality12b, F107_sel];
    [b_full, ~, ~, ~, ~] = regress(Y, X_full);
    percent_full = b_full(2) * 60 / sectionmean * 100;
    disp(['完整多元回归模型每十年增长百分比: ', num2str(percent_full), '%']);
    disp(' ');

    % 2. 简单线性回归（仅时间项）
    [b_simple, ~, ~, ~, ~] = regress(Y, [ones(xend,1), time]);
    percent_simple = b_simple(2) * 60 / sectionmean * 100;
    disp(['简单线性回归每十年增长百分比: ', num2str(percent_simple), '%']);
    disp(' ');

    % 3. 二级回归（已在上面计算）
    disp(['二级回归每十年增长百分比: ', num2str(percent_growth), '%']);
    disp(' ');

elseif section == 2
    % ====== 分支2：小样本模式（启用共线性诊断 + 岭回归 + 三模型对比） ======
    monthbegin = 103;
    monthend   = 120;
    disp('警告：小样本模式，启用共线性解决方案');

    xend = monthend - monthbegin + 1;  % 样本长度

    % 计算全局均值（后续计算百分比时使用）
    sectionmean = mean( ...
        mean( ...
            squeeze(mean(all_point_f(:,:,monthbegin:monthend), 1, 'omitnan')), ...
        'omitnan'), ...
    'omitnan');

    % 计算目标变量 Y
    meandh = squeeze(mean(mean(all_point_f(:,:,monthbegin:monthend), 1, 'omitnan'), 2, 'omitnan'))';
    meandh(meandh == 0) = NaN;
    Y = meandh(:);   % xend×1

    % 构造时间变量
    time = (1:xend)';   % xend×1

    % 构造 F107 指数：取奇数索引并去均值
    F107data_n = F107data - mean(F107data);
    F107data_f = NaN(120, 1);
    for n = 1:120
        F107data_f(n) = F107data_n(2*n - 1);
    end
    F107_sel = F107data_f(monthbegin:monthend);  % xend×1

    % 构造季节项（6 个月和 12 个月周期）
    seasonality6a  = cos(2*pi*2*time/6);
    seasonality6b  = sin(2*pi*2*time/6);
    seasonality12a = cos(2*pi*2*time/12);
    seasonality12b = sin(2*pi*2*time/12);

    % —— 共线性诊断 —— 
    X_full_diagnose = [ ...
        time, ...
        seasonality6a, seasonality6b, ...
        seasonality12a, seasonality12b, ...
        F107_sel ...
    ];  % xend×6

    % 条件数诊断
    cond_X = cond(X_full_diagnose);
    disp(['条件数: ', num2str(cond_X)]);
    if cond_X > 30
        disp('❌ 严重共线性问题 (条件数 > 30)');
    end

    % VIF 诊断（基于相关系数矩阵的逆对角线元素）
    vif_values = diag(inv(corrcoef(X_full_diagnose)));
    disp('方差膨胀因子 (VIF):');
    disp(array2table(vif_values', 'VariableNames', ...
        {'Time','S6a','S6b','S12a','S12b','F107'}));

    % 相关系数矩阵热图
    figure('Name','共线性诊断（小样本）');
    heatmap( corrcoef(X_full_diagnose), ...
        'Colormap', parula, ...
        'XData', {'Time','S6a','S6b','S12a','S12b','F107'}, ...
        'YData', {'Time','S6a','S6b','S12a','S12b','F107'} );
    title('变量相关系数矩阵');

    % —— 三种回归模型：简单线性、多元线性、岭回归 —— 

    % 1. 简单线性回归：Y ~ time
    [b_simple2, ~, residuals_simple2, ~, stats_simple2] = regress(Y, [ones(xend,1), time]);
    R2_simple2    = stats_simple2(1);
    percent_simple2 = b_simple2(2) * 60 / sectionmean * 100;

    % 2. 完整多元回归：Y ~ [time + 季节项 + F107]
    X_mult2 = [ ...
        ones(xend,1), ...
        time, ...
        seasonality6a, seasonality6b, ...
        seasonality12a, seasonality12b, ...
        F107_sel ...
    ];  % xend×7

    [b_full2, ~, residuals_full2, ~, stats_full2] = regress(Y, X_mult2);
    R2_full2       = stats_full2(1);
    percent_full2  = b_full2(2) * 60 / sectionmean * 100;

    % 3. 岭回归：Y ~ [time + 季节项 + F107]，带 λ 正则化
    lambda = 0.5;  % 正则化参数，可根据需要微调

    X_orig = [ ...
        time, ...
        seasonality6a, seasonality6b, ...
        seasonality12a, seasonality12b, ...
        F107_sel ...
    ];  % xend×6
    Y_orig = Y;  % xend×1

    % 手动标准化 X (6 列) 和 Y
    X_mean    = mean(X_orig, 1);     % 1×6
    X_std_val = std(X_orig,  0, 1);   % 1×6
    Y_mean    = mean(Y_orig);
    Y_std_val = std(Y_orig);

    X_scaled = [ ones(xend,1), (X_orig - X_mean) ./ X_std_val ];  % xend×7
    Y_scaled = (Y_orig - Y_mean) / Y_std_val;                    % xend×1

    b_ridge_scaled = ridge(Y_scaled, X_scaled(:, 2:end), lambda, 0);  % 7×1
    % b_ridge_scaled(1) 是标准化空间下的截距，后面 6 个是各变量系数

    % 反标准化：恢复原始量纲
    intercept_scaled = b_ridge_scaled(1);
    coeffs_scaled   = b_ridge_scaled(2:end);  % 6×1

    intercept_orig = Y_mean - sum( (X_mean ./ X_std_val) .* coeffs_scaled' ) * Y_std_val;
    coeffs_orig   = (coeffs_scaled ./ X_std_val') * Y_std_val;  % 6×1

    % 提取时间项系数
    time_coeff_ridge = coeffs_orig(1);

    % 计算岭回归每十年增长量与百分比
    decade_growth_ridge = time_coeff_ridge * 60;
    percent_growth_ridge = decade_growth_ridge / sectionmean * 100;

    % 计算岭回归拟合值与残差
    Y_ridge_fit     = intercept_orig + X_orig * coeffs_orig;  % xend×1
    residuals_ridge = Y_orig - Y_ridge_fit;                   % xend×1

    % 计算岭回归 R²
    R2_ridge = 1 - sum((Y_orig - Y_ridge_fit).^2) / sum((Y_orig - mean(Y_orig)).^2);

    % —— 残差正态性检验（岭回归） —— 
    [~, jb_pvalue2] = jbtest(residuals_ridge);
    % 若 xend 很小，jbtest 会给出 p=0.5，表示检验能力有限

    figure('Name','残差正态性诊断（小样本，岭回归）');
    subplot(1,3,1);
    qqplot(residuals_ridge);
    title(['Q-Q 图 (JB p=',num2str(jb_pvalue2,3),')']);
    grid on;

    subplot(1,3,2);
    histfit(residuals_ridge, 10);
    title('残差分布直方图');
    xlabel('残差值');
    ylabel('频数');

    subplot(1,3,3);
    pd2 = fitdist(residuals_ridge,'Kernel','Kernel','normal');
    x_vals2 = linspace(min(residuals_ridge), max(residuals_ridge), 100);
    pdf_vals2 = pdf(pd2, x_vals2);
    plot(x_vals2, pdf_vals2, 'LineWidth', 2);
    hold on;
    plot(x_vals2, normpdf(x_vals2, mean(residuals_ridge), std(residuals_ridge)), 'r--');
    title('核密度估计 vs 理论正态');
    legend('残差密度','理论正态','Location','best');

    % —— 可视化三种模型的拟合效果对比 —— 
    % 上图：原始 Y 与 各模型拟合值
    figure('Name','三种模型拟合对比（小样本）');
    subplot(3,1,1);
    plot(Y,            'b-', 'LineWidth', 1.5, 'DisplayName', '原始数据'); hold on;
    plot([1, xend], [0,0], 'k--', 'LineWidth', 1, 'DisplayName', '零参考线');
    title('原始数据（全体 Y）');
    xlabel('时间索引'); ylabel('变量值');
    legend('show');

    subplot(3,1,2);
    plot(time, Y,        'b-',  'LineWidth', 1.5, 'DisplayName', '原始数据'); hold on;
    plot(time, X_mult2 * b_full2, 'r-',  'LineWidth', 1.5, 'DisplayName', '完整多元回归拟合');
    plot(time, [ones(xend,1), time] * b_simple2, 'g--', 'LineWidth', 1, 'DisplayName', '简单线性回归拟合');
    legend('show');
    title('简单 vs 完整多元回归拟合曲线');
    xlabel('时间索引'); ylabel('变量值');

    subplot(3,1,3);
    plot(time, Y,            'b-', 'LineWidth', 1.5, 'DisplayName', '原始数据'); hold on;
    plot(time, Y_ridge_fit, 'm-', 'LineWidth', 1.5, 'DisplayName', '岭回归拟合');
    legend('show');
    title('岭回归拟合曲线');
    xlabel('时间索引'); ylabel('变量值');

    % —— 输出三种模型的统计结果 —— 
    disp('===== 小样本模式：三种模型结果对比 =====');

    % —— 简单线性回归 —— 
    disp('--- 简单线性回归 (Y ~ time) ---');
    disp(['  系数 b0: ', num2str(b_simple2(1)), ', b1: ', num2str(b_simple2(2))]);
    disp(['  R²: ', num2str(R2_simple2)]);
    disp(['  每十年增长百分比: ', num2str(percent_simple2), '%']);
    disp(' ');

    % —— 完整多元回归 —— 
    disp('--- 完整多元回归 (Y ~ time + 季节项 + F107) ---');
    disp(['  系数向量 [b0, b_time, b_S6a, b_S6b, b_S12a, b_S12b, b_F107]:']);
    disp(num2str(b_full2', '    %g'));
    disp(['  R²: ', num2str(R2_full2)]);
    disp(['  每十年增长百分比（使用 b_time）: ', num2str(percent_full2), '%']);
    disp(' ');

    % —— 岭回归 —— 
    disp('--- 岭回归 (带 λ=0.5 正则化) ---');
    disp(['  原始量纲下截距: ', num2str(intercept_orig)]);
    disp('  原始量纲下系数向量 [time, S6a, S6b, S12a, S12b, F107]:');
    disp(num2str(coeffs_orig', '    %g'));
    fprintf('  R² (岭回归): %.4f\n', R2_ridge);
    disp(['  每十年增长量: ', num2str(decade_growth_ridge)]);
    disp(['  每十年增长百分比: ', num2str(percent_growth_ridge), '%']);

    % —— 关于 JB 检验 p=0.5 的说明 —— 
    % 当样本量很小时（xend=18），jbtest 会返回 p=0.5（MATLAB 默认上限），
    % 表示在该非常小的样本下无法拒绝正态分布假设，但检验本身不够稳定，需谨慎解读。

    % —— 关于岭回归在小样本高共线性下的科学性 —— 
    % 1. 岭回归通过 L2 正则化项 λ||β||^2 来削弱共线性对系数估计的影响，能在 p 接近 n 或 n 很小时提供更稳定的系数。
    % 2. 由于小样本 R² 可能会因为过拟合而偏高，岭回归会适当收缩系数，通常能得到更可靠的预测性能。
    % 3. 可靠性依赖于 λ 的选择。这里示例取 λ=0.5，可通过交叉验证（CV）进一步调优以获得最优 λ。
    % 4. 若担心 R² 过于乐观，可额外计算交叉验证下的平均决定系数（CV R²）来评估模型泛化能力。

else
    error('输入无效，请输入 1 或 2');
end
