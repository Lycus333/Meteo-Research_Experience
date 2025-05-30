%%%ncdisp('G:\SABER_L2C\2014\SABER_L2C_2014100_Nprofs_1_02.0.nc');

%% 数据准备
clc;
clear;
% load('Trenddata.mat');
% load('F107data.mat');
% load('QBO30data.mat');
% load('QBO10data.mat');
% load('ENSOdata.mat');

varName ={'CO2retr','date','altitude','lon','lat'};
varnum=length(varName);

latticewid=2;
latlength=180/latticewid+1;
lonlength=360/latticewid+1;

latm=-50;
latn=50;
%%%%54
% %%选取高度 1对应140km 11对应30km


% 90-110km
top_s=31;
bottom_s=51;

% %65-110km
% top_s=31;
% bottom_s=76;

% %100km处（98-102）
% top_s=39;
% bottom_s=41;

dh=bottom_s-top_s+1;

all_point=NaN(latlength,dh,12,480);

for year=2002:2021
    %year=2002;
    datadir = ['G:\SABER_L2C\',num2str(year),'\'];
    filelist0 = dir(datadir);
    filelist0([1 2],:)=[];
    k=length(filelist0);
    point=NaN(lonlength,latlength,dh,k);
    
    month_cor=NaN(k,2);
    %%%读取一天的数据
    for d =1:k
        %d=8;
        filepath=strcat(datadir,filelist0(d).name);
        for i =1:varnum
            varstr1 = cell2mat(varName(i));
            var_nc = ncread(filepath,varstr1);
            eval([varstr1,'=var_nc;'])
        end
        
        CO2retr_int=CO2retr(top_s:bottom_s,:)';
        date_c=mod(mode(date),1000); %#ok<*DATE>
        
        %%%代码文件同文件夹getMonth
        [month,half]=getMonth(year,date_c);
        month_cor(d,1)=month;
        month_cor(d,2)=half;
        
        %    o=8;h=2;
        lonIdx = round(lon(:, 1) / latticewid) + 1;
        latIdx = round((lat(:, 1) + 90) / latticewid) + 1;
        
        for o=1:size(CO2retr_int,1)
            point(lonIdx(o),latIdx(o),:,d)= CO2retr_int(o,:);
        end
        point(point==0)=NaN;
        
    end
    
    monthValues = unique(month_cor);
    year_point = NaN(lonlength, latlength, dh, 24);
    for i = 1:length(monthValues)
        value = monthValues(i);
        half=1;
        indices = (month_cor(:,1) == value) & (month_cor(:,2) == half);
        year_point(:,:,:,2*value-1)=mean(point(:,:,:,indices),4,'omitnan');
        half=2;
        indices = (month_cor(:,1) == value) & (month_cor(:,2) == half);
        year_point(:,:,:,2*value)=mean(point(:,:,:,indices),4,'omitnan');
    end
    
    
    if (year==2002)
        all_point=squeeze(mean(year_point,1,'omitnan'));
    else
        all_point=cat(3,all_point,squeeze(mean(year_point,1,'omitnan')));
    end
    
    
end

all_point(all_point==0)=NaN;
all_point_f=NaN(size(all_point,1),size(all_point,2),120);

%%%双月平均得all_point_f
all_point_f(:,:,1)=mean(all_point(:,:,1:3),3,'omitnan');
all_point_f(:,:,120)=mean(all_point(:,:,476:479),3,'omitnan');
for f=2:119
all_point_f(:,:,f)=mean(all_point(:,:,(4*f-4):(4*f-1)),3,'omitnan');
end
all_point_f=all_point_f((180/(2*latticewid)+1+latm/latticewid):(180/(2*latticewid)+1+latn/latticewid),:,:);
all_point_f(all_point_f==0)=NaN;
%数据修正
all_point_f(4,:,74)=mean(all_point_f([3 5],:,74),1,'omitnan');
all_point_f(45,:,59)=mean(all_point_f([44 46],:,59),1,'omitnan');
all_point_f(7,:,97)=mean(all_point_f([6 8],:,97),1,'omitnan');
all_point_f(5,:,22)=mean(all_point_f([4 6],:,22),1,'omitnan');

sectionmean=mean(mean(squeeze(mean(all_point_f,'omitnan')),'omitnan'),'omitnan');%(:,:,monthbegin:monthend)

%%%%%存数据

% for i=1:size(all_point_f,2)
%     for j=1:size(all_point_f,3)
%         all_point_smooth=smoothdata(squeeze(all_point_f(:,i,j)), 'movmean', 20);
%     end
% end
% 
% all_point_smooth=nan(51,21,120);
% for i=1:size(all_point_f,1)
%     for j=1:size(all_point_f,2)
%         all_point_smooth(i,j,:)=smoothdata(squeeze(all_point_f(i,j,:)), 'movmean', 2);
%     end
% end


latgrid=latm:latticewid:latn;
height=nan(dh,1);
for h=1:dh
    height(h,1)=111-h;
end

%%%%colormap的设置
colors=[109 001 031;183 034 048;220 109 087;246 178 147;251 227 213;
    233 241 244;182 215 232;109 173 209;049 124 183;016 070 128;];
colors=colors/255;
originalNumColors = size(colors, 1);
newColors = 40;
step = (originalNumColors - 1) / (newColors - 1);
interpIndices = 1:step:originalNumColors;
interpColors = interp1(1:originalNumColors, colors, interpIndices);
colors=flipud(interpColors);

figure;
camax=max(max(max(all_point_f)));
camin=min(min(min(all_point_f)));
for i=1:120
    
contourf(latgrid,height,all_point_f(:,:,i)',30,'linecolor','none');
set(gca,'YDir','normal');
ylabel('altitude','fontname','Times New Roman','fontsize',16);
xlabel('latitude','fontname','Times New Roman','fontsize',16);
title(num2str(i),'fontsize',17,'fontname','Times New Roman');
colormap(colors);
colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6]);
caxis= ([camin camax]);
imageFileName = [num2str(i),'.png'];
folderPath = 'C:\Users\UNGIFTED\Desktop\CO2变化及其冷却作用研究\CO2 VMR\24.3.20\90-110km\120';
saveas(gca, fullfile(folderPath, imageFileName));
% pause;

end


% 
% numImages = 120;
% folderPath = 'C:\Users\UNGIFTED\Desktop\CO2变化及其冷却作用研究\CO2VMR\24.3.20\90-110km\120';
% gifFileName = '120.gif';
% delayTime = 0.15;
% images = cell(1, 3);
% 
% % 创建一个进度条
% h = waitbar(0, 'Creating GIF...');
% 
% % 读取每个通道的图片并存储到 cell 数组中
% for channel = 1:3
%     channelImages = cell(1, numImages);
%     for i = 1:numImages
%         % 更新进度条
%         progress = (channel - 1) * numImages + i;
%         waitbar(progress / (3 * numImages), h, sprintf('Creating GIF... %d%%', round(progress / (3 * numImages) * 100)));
% 
%         % 生成图片文件名
%         imageFileName = fullfile(folderPath, [num2str(i) '.png']);
%         % 读取图片
%         img = imread(imageFileName);
%         % 取出当前通道的图片数据
%         channelImage = img(:,:,channel);
%         channelImages{i} = channelImage;
%     end
%     % 将当前通道的图片数据存入总的图片数据数组中
%     images{channel} = channelImages;
% end
% 
% % 关闭进度条
% close(h);
% 
% % 合并三个通道的GIF为一个GIF
% h = waitbar(0, 'Writing GIF...');
% 
% for i = 1:numImages
%     % 更新进度条
%     waitbar(i / numImages, h, sprintf('Writing GIF... %d%%', round(i / numImages * 100)));
% 
%     % 将第一张图片写入 GIF
%     if i == 1
%         % 读取每个通道的第一张 GIF 图像
%         redImage = imread('temp_1.gif');
%         greenImage = imread('temp_2.gif');
%         blueImage = imread('temp_3.gif');
%         % 合并为一个 GIF
%         imwrite(redImage, gifFileName, 'gif', 'Loopcount', inf, 'DelayTime', delayTime);
%         imwrite(greenImage, gifFileName, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
%         imwrite(blueImage, gifFileName, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
%     else
%         % 追加后续图片到 GIF
%         % 读取每个通道的当前 GIF 图像
%         redImage = imread('temp_1.gif', 'Index', i);
%         greenImage = imread('temp_2.gif', 'Index', i);
%         blueImage = imread('temp_3.gif', 'Index', i);
%         % 将每个通道的当前 GIF 图像追加到 GIF
%         imwrite(redImage, gifFileName, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
%         imwrite(greenImage, gifFileName, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
%         imwrite(blueImage, gifFileName, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
%     end
% end
% 
% % 关闭进度条
% close(h);
% 
% % 删除临时文件
% delete('temp_1.gif', 'temp_2.gif', 'temp_3.gif');

%%

%%% 2-102  103-120
monthbegin = 2;
monthend = 102;
% 
% monthbegin = 103;monthend = 120;
X = [ones(xend,1), time];  % 设计矩阵（含常数项）
Y = meandh(monthbegin-102:monthend-102)'; % 因变量
[b, ~, ~, ~, stats] = regress(Y, X);

decade_growth = b(2) * 60;  % 正确性验证：双月单位 ×60=10年
percent_growth = decade_growth / sectionmean * 100; 


%% 增长率&相关系数
%%% t的取值，是否要带三角函数周期
%%% sectionmean的选择，尽管影响不大
%%% 秩亏，舍弃2、3月周期
clc;
clear;
load('25.5.30.mat');

% load('Trenddata.mat');
load('F107data.mat');
load('QBO30data.mat');
load('QBO10data.mat');
load('ENSOdata.mat');

%%% 2-102  103-120
monthbegin = 2;
monthend = 102;
% 
monthbegin = 103;monthend = 120;
%2019-2021
xend=monthend-monthbegin+1;

meandh=squeeze(mean(mean(all_point_f(:,:,monthbegin:monthend),1,'omitnan'),2,'omitnan'))';
meandh(meandh==0)=NaN;

time=(1:xend)'; %#ok<*NASGU>

% 使用线性回归模型
QBO30data_n=table2array(QBO30data);
QBO30data_n=reshape(QBO30data_n.',[],1);
QBO10data_n=table2array(QBO10data);
QBO10data_n=reshape(QBO10data_n.',[],1);
F107data_n=F107data;

QBO30data_f=NaN(120,1);
QBO10data_f=NaN(120,1);
F107data_f=NaN(120,1);
for n=1:120
    QBO30data_f(n,1)=QBO30data_n(2*n-1,1);
end
for n=1:120
    QBO10data_f(n,1)=QBO10data_n(2*n-1,1);
end
for n=1:120
    F107data_f(n,1)=F107data_n(2*n-1,1);
end
QBO30data_f=QBO30data_f/10;
QBO10data_f=QBO10data_f/10;

ENSOdata_f=ENSOdata(2:2:280);
ENSOdata_f(7:7:140)=[];

% Trenddata_f=Trenddata(1:2:240);

time=(1:xend)';
seasonality3a=cos(2*pi*2*time/3);
seasonality3b=sin(2*pi*2*time/3);
seasonality4a=cos(2*pi*2*time/4);
seasonality4b=sin(2*pi*2*time/4);
seasonality6a=cos(2*pi*2*time/6);
seasonality6b=sin(2*pi*2*time/6);
seasonality12a=cos(2*pi*2*time/12);
seasonality12b=sin(2*pi*2*time/12);

t=[time];
% , seasonality6a.*time,seasonality6b.*time, seasonality12a.*time,seasonality12b.*time

% QBO30=[QBO30data_f(monthbegin:monthend), seasonality6a.*QBO30data_f(monthbegin:monthend),...
%     seasonality6b.*QBO30data_f(monthbegin:monthend), seasonality12a.*QBO30data_f(monthbegin:monthend),...
%     seasonality12b.*QBO30data_f(monthbegin:monthend)];
% 
% QBO10=[QBO10data_f(monthbegin:monthend), seasonality6a.*QBO10data_f(monthbegin:monthend),...
%     seasonality6b.*QBO10data_f(monthbegin:monthend), seasonality12a.*QBO10data_f(monthbegin:monthend),...
%     seasonality12b.*QBO10data_f(monthbegin:monthend)];

F107=[F107data_f(monthbegin:monthend), seasonality6a.*F107data_f(monthbegin:monthend),...
    seasonality6b.*F107data_f(monthbegin:monthend), seasonality12a.*F107data_f(monthbegin:monthend),...
    seasonality12b.*F107data_f(monthbegin:monthend)];
% 
% ENSO=[ENSOdata_f(monthbegin:monthend), seasonality6a.*ENSOdata_f(monthbegin:monthend),...
%     seasonality6b.*ENSOdata_f(monthbegin:monthend), seasonality12a.*ENSOdata_f(monthbegin:monthend),...
%     seasonality12b.*ENSOdata_f(monthbegin:monthend)];

X = [ones(xend, 1), t, ...
    seasonality6a, seasonality6b, seasonality12a, seasonality12b, ...
    F107];%    seasonality3a, seasonality3b, seasonality4a, seasonality4b, ...        
Xnum=size(X,2);


Y=meandh';
[b,bint,r,rint,stats]=regress(Y(1:xend,:),X(1:xend,:));
sectionmean=mean(mean(squeeze(mean(all_point_f(:,:,monthbegin:monthend),'omitnan')),'omitnan'),'omitnan');%
b_ave=NaN(125,1);
for i=1:xend
    b_ave(i,1)=b(2);%+b(3)*cos(2*pi*2*i/6)+b(4)*sin(2*pi*2*i/6)+b(5)*cos(2*pi*2*i/12)+b(6)*sin(2*pi*2*i/12);
end
percent=mean(b_ave,'omitnan')*60 / sectionmean *100;
disp(['每十年平均增长率：', num2str(mean(b_ave,'omitnan')*60)]);
disp(['每十年增长百分比：', num2str(percent),'%']);

figure;
plot(Y);
hold on
fit_a=NaN(xend,Xnum);
Yfit=NaN(1,xend);

for i=1:xend
    fit_a(i,:)=X(i,:).*b(:)';
    Yfit(i)=sum(fit_a(i,:));
end
plot(Yfit);

xiangguanxishu=corrcoef(Yfit,Y);
disp(['相关系数：', num2str(xiangguanxishu(1,2))]);
disp(['区间平均',num2str(sectionmean)]);



disp(' ');
X = [ones(xend,1), time];  % 设计矩阵（含常数项）
Y = meandh(1:end)'; % 因变量
[b, ~, ~, ~, stats] = regress(Y, X);

decade_growth = b(2) * 60;  % 正确性验证：双月单位 ×60=10年
percent_growth = decade_growth / sectionmean * 100; 
disp(['仅仅线性拟合：',num2str(percent_growth),'%']);

%%




%solar
%        response_coefficients(w,h)=b(11)*100/nanmean(all_point_f(w,h,,monthbegin:monthend));
%线性增长
%        response_coefficients(w,h)=b(2)*60/nanmean(all_point_f(w,h,,monthbegin:monthend));

response_coefficients =NaN(size(all_point_f,1),size(all_point_f,2));
correlation_matrix=NaN(size(all_point_f,1),size(all_point_f,2));
Yfit=NaN(1,xend);

for w=1:size(all_point_f,1)
    for h=1:size(all_point_f,2)
        Y=squeeze(all_point_f(w,h,monthbegin:monthend));
%         Y=fillmissing(Y,'linear');%%%?????
        [b,bint,r,rint,stats]=regress(Y(1:xend,:),X(1:xend,:));
        for i=1:xend
            for j=1:Xnum
                fit_a(1,j)=X(i,j)*b(j);
            end
            Yfit(1,i)=sum(fit_a(1:j));
            fit_a=[];
        end
        correlation=corrcoef(Yfit',Y);
        correlation_matrix(w,h)=correlation(1,2);
        for i=1:xend
            b_ave(i,1)=b(2)+b(3)*cos(2*pi*2*i/6)+b(4)*sin(2*pi*2*i/6)+b(5)*cos(2*pi*2*i/12)+b(6)*sin(2*pi*2*i/12);
        end
        response_coefficients(w,h)=mean(b_ave,'omitnan')*60/mean(all_point_f(w,h,monthbegin:monthend),'omitnan');%
    end
end

latgrid=latm:latticewid:latn;
height=nan(dh,1);
for h=1:dh
    height(h,1)=111-h;
end

%%%%colormap的设置
colors=[109 001 031;183 034 048;220 109 087;246 178 147;251 227 213;
    233 241 244;182 215 232;109 173 209;049 124 183;016 070 128;];
colors=colors/255;
originalNumColors = size(colors, 1);
newColors = 40;
step = (originalNumColors - 1) / (newColors - 1);
interpIndices = 1:step:originalNumColors;
interpColors = interp1(1:originalNumColors, colors, interpIndices);
colors=flipud(interpColors);

figure;
contourf(latgrid,height,response_coefficients',10,'linecolor','none');
set(gca,'YDir','normal');
ylabel('altitude','fontname','Times New Roman','fontsize',16);
xlabel('latitude','fontname','Times New Roman','fontsize',16);
titlename='growth rate';
title(titlename,'fontsize',17,'fontname','Times New Roman');
% cmax=max(max(Nspring_data));
% cmin=min(min(Nspring_data));
% caxisRange = max(abs(cmin), abs(cmax));
% if (monthend==102)
%     caxis([0.04, 0.085]);
% else
%     caxis([0, 0.4]);
% end
colormap(colors);
colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6],...
    'TickLabels', sprintfc('%.1f%%', get(colorbar, 'Ticks')*100));


figure;
contourf(latgrid,height,correlation_matrix',10,'linecolor','none');
set(gca,'YDir','normal');
ylabel('altitude','fontname','Times New Roman','fontsize',16);
xlabel('latitude','fontname','Times New Roman','fontsize',16);
titlename='correlation coefficient';
title(titlename,'fontsize',17,'fontname','Times New Roman');
% cmax=max(max(Nspring_data));
% cmin=min(min(Nspring_data));
% caxisRange = max(abs(cmin), abs(cmax));
colormap(colors);
colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6],...
    'TickLabels', sprintfc('%.2f', get(colorbar, 'Ticks')));


 







%% 省略
%动态回归
%{
% 计算自相关函数
vec_X=X(:);
lags =36; 
[acf, lags] = autocorr(vec_X, lags);

stem(lags, acf);
xlabel('Lags');
ylabel('Autocorrelation');
title('Autocorrelation Function (ACF)');


lags = 5; % 滞后期数
X_lags = lagmatrix(X(1:110), 1:lags);
X1=[X(1:110,:) X_lags];

Mdl = fitlm(X1,Y(1:110));

disp('Dynamic Regression Model Coefficients:');
disp(Mdl.Coefficients);

Y_new =cat(1,nan(10,1) ,predict(Mdl, X1));

plot(time, Y, '-b');
hold on;
plot(time, Y_new, '-r')
xlabel('Time');
ylabel('Value');
legend('Original Data', 'Forecast');

%}
%%%使用神经网络拟合
%{
clc;
clear;
load('Trenddata.mat');
load('F107data.mat');
load('QBO30data.mat');
load('QBO10data.mat');
load('ENSOdata.mat');

%%% 1-102  103-120
monthbegin = 1;
monthend = 120;

% monthbegin = 103;monthend = 120;
% monthbegin = 103;monthend = 120;
xend=monthend-monthbegin+1;

meandh=squeeze(nanmean(nanmean(all_point_f(:,:,monthbegin:monthend),1),2))';
meandh(meandh==0)=NaN;

QBO30data_n=table2array(QBO30data);
QBO30data_n=reshape(QBO30data_n.',[],1);
QBO10data_n=table2array(QBO10data);
QBO10data_n=reshape(QBO10data_n.',[],1);
F107data_n=F107data;

QBO30data_f=NaN(120,1);
QBO10data_f=NaN(120,1);
F107data_f=NaN(120,1);
for n=1:120
QBO30data_f(n,1)=QBO30data_n(2*n-1,1);
end
for n=1:120
QBO10data_f(n,1)=QBO10data_n(2*n-1,1);
end
for n=1:120
F107data_f(n,1)=F107data_n(2*n-1,1);
end
QBO30data_f=QBO30data_f/10;
QBO10data_f=QBO10data_f/10;

ENSOdata_f=ENSOdata(1:2:279);
ENSOdata_f(1:7:134)=[];

Trenddata_f=Trenddata(1:2:240);

time=(1:xend)';
seasonality3a=cos(2*pi*time/6);
seasonality3b=sin(2*pi*time/6);
seasonality4a=cos(2*pi*time/8);
seasonality4b=sin(2*pi*time/8);
seasonality6a=cos(2*pi*time/12);
seasonality6b=sin(2*pi*time/12);
seasonality12a=cos(2*pi*time/24);
seasonality12b=sin(2*pi*time/24);
X=[ones(xend,1),time,seasonality3a,seasonality3b,seasonality4a,seasonality4b,...
    seasonality6a,seasonality6b,seasonality12a,seasonality12b,...
    QBO30data_f(monthbegin:monthend),QBO10data_f(monthbegin:monthend),...
    F107data_f(monthbegin:monthend),ENSOdata_f(monthbegin:monthend)  ];%14  ,Trenddata_f(monthbegin:monthend)
Y=meandh';


% 数据预处理
data_size = length(Y);
train_size = floor(0.9 * data_size); % 80% 的数据作为训练集
X_train = X(18:train_size, :);
Y_train = Y(18:train_size);
X_test = X(train_size+1:end, :);
Y_test = Y(train_size+1:end);

% 将输入数据转换为元胞数组
X_train_cell = cell(size(X_train, 1), 1);
X_test_cell = cell(size(X_test, 1), 1);
for i = 1:size(X_train, 1)
    X_train_cell{i} = X_train(i, :)';
end
for i = 1:size(X_test, 1)
    X_test_cell{i} = X_test(i, :)';
end

% 构建 RNN 模型
input_size = size(X, 2);
hidden_size = 600; % 隐藏层大小
output_size = 1;
layers = [ ...
    sequenceInputLayer(input_size)
    lstmLayer(hidden_size, 'OutputMode', 'last')
    fullyConnectedLayer(output_size)
    regressionLayer];

% 设置训练选项
options = trainingOptions('adam', 'MaxEpochs', 700, 'MiniBatchSize', 32, ...
    'GradientThreshold', 0.000001, 'Shuffle', 'never', 'Verbose', 0);

% 训练 RNN 模型
net = trainNetwork(X_train_cell, Y_train, layers, options);

% 使用训练好的模型进行预测
Y_pre = predict(net, X_test_cell);

% 计算模型性能
mse = mean((Y_test - Y_pre').^2);
fprintf('均方误差 (MSE): %.4f\n', mse);

figure;
plot(meandh);
hold on
plot(Y_pre);



% 归一化处理
[Xn, PS_X] = mapminmax(X',-1,1);
[Yn, PS_Y] = mapminmax(Y',-1,1);

cv = cvpartition(size(Xn, 1), 'KFold', 8);  % 8 折交叉验证
hiddenLayerSizeRange = [1:20];  % 隐藏层节点数范围
trainFcnList = {'trainlm', 'trainbfg', 'trainrp'};  % 训练函数列表
% 训练方法
%{
1. **'trainlm' (Levenberg-Marquardt 算法)**
   - 特点：使用 Levenberg-Marquardt 算法进行网络的训练，是一种快速且有效的训练算法，通常用于小型数据集和中等规模的神经网络。
   - 适用场景：适用于较小的数据集和中等规模的神经网络，能够在较短的时间内得到较好的训练效果。

2. **'traingd' (梯度下降算法)**
   - 特点：使用标准的梯度下降算法进行网络的训练，每次迭代都根据梯度更新网络权重，可能会出现在局部最优解中。
   - 适用场景：适用于较大的数据集和复杂的神经网络，虽然训练速度较慢，但有可能获得更好的全局最优解。

3. **'traingda' (自适应梯度下降算法)**
   - 特点：自适应地调整学习率，根据梯度大小自动调整学习率，能够加速训练过程并提高训练的稳定性。
   - 适用场景：适用于大型数据集和复杂的神经网络，能够加快训练速度并提高训练的稳定性。

4. **'traingdx' (自适应梯度下降算法，带动量项)**
   - 特点：在自适应梯度下降算法的基础上添加了动量项，能够加速训练过程并提高训练的稳定性。
   - 适用场景：适用于大型数据集和复杂的神经网络，能够加快训练速度、提高训练的稳定性，并可能获得更好的全局最优解。

5. **'trainrp' (逆向传播算法)**
   - 特点：使用逆向传播算法进行网络的训练，采用批量更新权重的方式，可以提高训练的速度和稳定性。
   - 适用场景：适用于中等规模的数据集和神经网络，能够获得较好的训练效果，并且训练速度较快。

6. **'trainbfg' (BFGS 优化算法)**
   - 特点：使用 BFGS 优化算法进行网络的训练，是一种高效的二阶优化算法，通常用于小型数据集和中等规模的神经网络。
   - 适用场景：适用于小型数据集和中等规模的神经网络，能够在较短的时间内得到较好的训练效果。

%}
bestPerformance = Inf;
bestNet = [];

% 交叉验证调参
for i = 1:numel(hiddenLayerSizeRange)
    for j = 1:numel(trainFcnList)
        hiddenLayerSize = hiddenLayerSizeRange(i);
        trainFcn = trainFcnList{j};
        net = newff(Xn, Yn, [50,hiddenLayerSize], {'tansig', 'purelin'}, trainFcn, 'mse');% 创建神经网络
        performance = zeros(1, cv.NumTestSets);% 交叉验证
        for k = 1:cv.NumTestSets
            % 训练集和测试集划分
            trainIdx = cv.training(k);
            testIdx = cv.test(k);
            net = train(net, Xn(:, trainIdx), Yn(trainIdx));% 训练神经网络
            Y_pred = sim(net, Xn(:, testIdx));% 测试性能
            performance(k) = mse(Y_pred - Yn(testIdx));
        end
        avgPerformance = mean(performance);% 计算平均性能
        % 保存最佳模型
        if avgPerformance < bestPerformance
            bestPerformance = avgPerformance;
            bestNet = net;
        end
    end
end
disp(bestNet);

[net,tr,Y1,E]=train(bestNet,Xn,Yn);
Y_pre=mapminmax('reverse',Y1',PS_Y);

figure;
plot(meandh);
hold on
plot(Y_pre);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%归一化处理
[Xn,PS_X]=mapminmax(X',-1,1);
[Yn,PS_Y]=mapminmax(Y',-1,1);
net=newff(Xn,Yn,[50,5],{'tansig','purelin'},'trainbfg','mse','msereg');

net.trainFcn = 'trainlm';  % 使用 Levenberg-Marquardt 算法
net.trainParam.epochs = 5000;        %最大训练轮数
net.trainParam.min_grad = 1e-8;     %梯度下降的最小梯度
net.trainParam.max_fail = 5;       %允许连续验证误差不改善的次数
net.trainParam.weightRegularization = 0.01;  %L2正则化参数

[net,tr,Y1,E]=train(net,Xn,Yn);
Y_pre=mapminmax('reverse',Y1',PS_Y);
figure;
plot(meandh);
hold on
plot(Y_pre);

weights_input_hidden=net.Iw{1};
plot(weights_input_hidden);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%}
%%%%%%%以下是全球增长率的等高线图
%%%%经度*纬度（高度平均）
%{
clc;
clear;
%%%使用3分度值 年单位

yearbegin = 2019;
yearend = 2021;
yearnum=yearend-yearbegin+1;
yearrank=[];
for yr=1:yearnum
    yearrank(1,yr)=yr+yearbegin-1;
end

[X, Y] = meshgrid(lon, lat);

% 创建一个存储年增长率的矩阵，大小为[n, m]
growth_rate = zeros(size(all_point, 1), size(all_point, 2));

% 计算年增长率
for i = 1:size(all_point, 1)
    for j = 1:size(all_point, 2)
        % 使用线性拟合计算斜率
        x = yearrank;
        y = squeeze(all_point(i, j, yearbegin-2001:yearend-2001));
        coefficients = polyfit(x, y, 1);
        slope = coefficients(1);
        
        % 计算年增长率
        mean_value = mean(y);
        growth_rate(i, j) = slope / mean_value;
    end
end
growth_rate=growth_rate*10;

longrid_f=-180:latticewid:180;
%%%%colormap的设置
colors=[109 001 031;183 034 048;220 109 087;246 178 147;251 227 213;
    233 241 244;182 215 232;109 173 209;049 124 183;016 070 128;];
colors=colors/255;
originalNumColors = size(colors, 1);
newColors = 40;
step = (originalNumColors - 1) / (newColors - 1);
interpIndices = 1:step:originalNumColors;
interpColors = interp1(1:originalNumColors, colors, interpIndices);
colors=flipud(interpColors);

figure;
contourf(longrid_f,latgrid',growth_rate,400,'linecolor','none');
set(gca,'YDir','normal');
ylabel('latitude','fontname','Times New Roman','fontsize',16);
xlabel('longitude','fontname','Times New Roman','fontsize',16);
titlename=horzcat(num2str(yearbegin),'-',num2str(yearend),' 90-110km growth rate mean');
title(titlename,'fontsize',17,'fontname','Times New Roman');
% cmax=max(max(eofn));
% cmin=min(min(eofn));
% caxisRange = max(abs(cmin), abs(cmax));
caxis([-0.6, 1.3]);
colormap(colors);
colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6],...
    'TickLabels', sprintfc('%.1f%%', get(colorbar, 'Ticks')*100));
hold on
land=shaperead('landareas.shp','UseGeoCoords',true);
geoshow(land,'facecolor','none','edgecolor','black','linewidth',1);
hold off

% X = reshape(1:numel(growth_rate), size(growth_rate));
% Y = growth_rate(:);
% model = fitlm(X, Y);
% predicted_growth_rate = predict(model, X);
% predicted_growth_rate = reshape(predicted_growth_rate, size(growth_rate));
% [X, Y] = meshgrid(1:size(predicted_growth_rate, 2), 1:size(predicted_growth_rate, 1));
% contour(X, Y, predicted_growth_rate);
% 

%%%使用1分度值 季度单位
clc
clear
growth_rate = diff(all_point_month, 1, 3) ./ all_point_month(:, :, 1:end-1);
growth_rate1=squeeze(nanmean(growth_rate,2));


%}

%% 分布
%%%%纬度*高度（经度平均）分布 上面的是增长率
%
clc;
clear;


latticewid=2;
latgrid=-54:latticewid:54;
height=nan(dh,1);
for h=1:dh
height(h,1)=110-h;
end

%%%%colormap的设置
colors=[109 001 031;183 034 048;220 109 087;246 178 147;251 227 213;
    233 241 244;182 215 232;109 173 209;049 124 183;016 070 128;];
colors=colors/255;
originalNumColors = size(colors, 1);
newColors = 40;
step = (originalNumColors - 1) / (newColors - 1);
interpIndices = 1:step:originalNumColors;
interpColors = interp1(1:originalNumColors, colors, interpIndices);
colors=flipud(interpColors);

%%% 2-102  103-120
monthbegin = 2;
monthend = 102;
%  monthbegin = 91;monthend = 120; 
%2017-2021

all_point_f_average=mean(all_point_f(:,:,monthbegin:monthend),3,'omitnan');

figure;
contourf(latgrid,height,all_point_f_average',400,'linecolor','none');
set(gca,'YDir','normal');
ylabel('altitude','fontname','Times New Roman','fontsize',16);
xlabel('latitude','fontname','Times New Roman','fontsize',16);
titlename='65-110km CO2VMR Distribution';
title(titlename,'fontsize',17,'fontname','Times New Roman');
% cmax=max(max(Nspring_data));
% cmin=min(min(Nspring_data));
% caxisRange = max(abs(cmin), abs(cmax));
caxis([0.0001, 0.0004]);
colormap(colors);
colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6]);


%% 对分布 all_point_f（120）做eof分析
%%%为了显示的完整性，舍弃all_point_f(:,:,1);
clc;
clear;
load('Trenddata.mat');
load('F107data.mat');
load('QBO30data.mat');
load('QBO10data.mat');
load('ENSOdata.mat');

%%%%colormap的设置
colors=[109 001 031;183 034 048;220 109 087;246 178 147;251 227 213;
    233 241 244;182 215 232;109 173 209;049 124 183;016 070 128;];
colors=colors/255;
originalNumColors = size(colors, 1);
newColors = 40;
step = (originalNumColors - 1) / (newColors - 1);
interpIndices = 1:step:originalNumColors;
interpColors = interp1(1:originalNumColors, colors, interpIndices);
colors=flipud(interpColors);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   02-18年
monthbegin = 2;
monthend = 102;
xend=monthend-monthbegin+1;


height=nan(dh,1);
for h=1:dh
height(h,1)=110-h;
end

latgrid=-50:latticewid:50;
lata=latgrid';

all_point_f0=NaN(size(all_point_f));
for i=monthbegin:monthend
all_point_f0(:,:,i)=all_point_f(:,:,i).*(sqrt(cosd(lata)));
end
all_point_f0(all_point_f0==0)=NaN;

isEmptyMatrix = false(1, size(all_point_f0, 3));
for i = 1:size(all_point_f0, 3)
    isEmptyMatrix(i) = isnan(all_point_f0(:,:,i));
end
all_point_f0 = all_point_f0(:,:,~isEmptyMatrix);


[eof_maps,pc,expvar] = eof(all_point_f0(:,:,2:102));
eof1=-eof_maps(:,:,1).*std(pc(1,:));
pc1=-pc(1,:)./std(pc(1,:));
eof2=-eof_maps(:,:,2).*std(pc(2,:));
pc2=-pc(2,:)./std(pc(2,:));

eof3=-eof_maps(:,:,3).*std(pc(3,:));
pc3=-pc(3,:)./std(pc(3,:));
eof4=-eof_maps(:,:,4).*std(pc(4,:));
pc4=-pc(4,:)./std(pc(4,:));

eof5=-eof_maps(:,:,5).*std(pc(5,:));
pc5=-pc(5,:)./std(pc(5,:));
eof6=-eof_maps(:,:,6).*std(pc(6,:));
pc6=-pc(6,:)./std(pc(6,:));

%%%%%%%%%%%% eof
model=2;

modelname=['eof' num2str(model)];
eofn=eval(modelname); 
figure;
contourf(latgrid,height,eofn',200,'linecolor','none');
set(gca,'YDir','normal');
ylabel('latitude','fontname','Times New Roman','fontsize',16);
xlabel('longitude','fontname','Times New Roman','fontsize',16);
titlename=horzcat(modelname,'  ','(' ,num2str(expvar(model)),'%',')');
title(titlename,'fontsize',17,'fontname','Times New Roman');
colormap(colors);
c=colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6]);
caxis([-1.8e-5, 1.8e-5]);
c.Ticks= (-1.8:0.4:1.8) * 1e-5; 
% c.TickLabels=arrayfun(@(x) sprintf('%.0e', x), ticks, 'UniformOutput', false);
% c.Label.String = 'Intensity';
% c.Label.FontSize = 12;


%%%%%%%%%%%% pc

modelname=['pc' num2str(model)];
pcn=eval(modelname); 

figure;
plot([NaN,pcn],'o-','markerfacecolor','black','linewidth',1,'color','black');
cmax=max(max(pcn));
cmin=min(min(pcn));
caxisRange = max(abs(cmin), abs(cmax));
ylim([-caxisRange, caxisRange]);
set(gca,'YDir','normal');
xticks(1:6:xend);
xticklabels(2002:2021);
xtickangle(45);
ax=gca;
ax.XAxis.LineWidth=1;
xlabel('year','fontname','Times New Roman','fontsize',16);%绘制子图x标签
title(modelname,'fontname','Times New Roman','fontsize',17');
hold on
x=[1 xend];
xlim([x(1) x(end)]);
y=[0 0];
plot(x,y,'k--','linewidth',0.8);
hold off


%%%pc合图
figure;
bx=tight_subplot(4,1,[0.003 0.003],[0.1 0.1],[0.1 0.05]);
for pic=1:4
    axes(bx(pic));        %#ok<*LAXES>
    for model=1:4
        modelname=['pc' num2str(model)];
        pcn=eval(modelname);
        if model==pic
            plot([NaN,pcn],'linewidth',2,'displayname',modelname);
            text(-5, 0.5, modelname, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 0,...
                'fontname','Times New Roman','fontsize',15);
        else
            plot([NaN,pcn],'--','linewidth',1,'displayname',modelname);
        end
        box off;
        grid on;
        cmax=max(max(pcn));
        cmin=min(min(pcn));
        caxisRange = max(abs(cmin), abs(cmax));
        ylim([-caxisRange, caxisRange]);
        set(gca,'YDir','normal');
        if pic==4
            xticks(1:6:102);
            xticklabels(2002:2021);
            xtickangle(45);        
            xlabel('year','fontname','Times New Roman','fontsize',16);%绘制子图x标签
        else
            xticks(1:6:102);
        end
        hold on
    end
    
    x=[1 102];
    xlim([x(1) x(end)]);
    y=[0 0];
    plot(x,y,'k--','linewidth',0.8);
    hold off
end
sgtitle('pcs','fontname','Times New Roman','fontsize',17);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      19-21年
monthbegin=103;
monthend=120;
xend=monthend-monthbegin+1;

height=nan(dh,1);
for h=1:dh
height(h,1)=110-h;
end

latgrid=-50:latticewid:50;
lata=latgrid';

all_point_f0=NaN(size(all_point_f));

for i=monthbegin:monthend
all_point_f0(:,:,i)=all_point_f(:,:,i).*(sqrt(cosd(lata)));
end
all_point_f0(all_point_f0==0)=NaN;

isEmptyMatrix = false(1, size(all_point_f0, 3));
for i = 1:size(all_point_f0, 3)
    isEmptyMatrix(i) = all(all(isnan(all_point_f0(:,:,i))));
end
all_point_f0 = all_point_f0(:,:,~isEmptyMatrix);


[eof_maps,pc,expvar] = eof(all_point_f0(:,:,1:18));
eof1=eof_maps(:,:,1).*std(pc(1,:));
pc1=pc(1,:)./std(pc(1,:));
eof2=-eof_maps(:,:,2).*std(pc(2,:));
pc2=-pc(2,:)./std(pc(2,:));

eof3=eof_maps(:,:,3).*std(pc(3,:));
pc3=pc(3,:)./std(pc(3,:));
eof4=eof_maps(:,:,4).*std(pc(4,:));
pc4=pc(4,:)./std(pc(4,:));

eof5=-eof_maps(:,:,5).*std(pc(5,:));
pc5=-pc(5,:)./std(pc(5,:));
eof6=-eof_maps(:,:,6).*std(pc(6,:));
pc6=-pc(6,:)./std(pc(6,:));

%%%%%%%%%%%% eof
model=5;

modelname=['eof' num2str(model)];
eofn=eval(modelname); 
figure;
contourf(latgrid,height,eofn',200,'linecolor','none');
set(gca,'YDir','normal');
ylabel('latitude','fontname','Times New Roman','fontsize',16);
xlabel('longitude','fontname','Times New Roman','fontsize',16);
%%'position',[185,-90,0]
titlename=horzcat(modelname,'  ','(' ,num2str(expvar(model)),'%',')');
title(titlename,'fontsize',17,'fontname','Times New Roman');
colormap(colors);
c=colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6]);
caxis([-1.8e-5, 1.8e-5]);
c.Ticks= (-1.8:0.4:1.8) * 1e-5; 
% c.TickLabels=arrayfun(@(x) sprintf('%.0e', x), ticks, 'UniformOutput', false);
% c.Label.String = 'Intensity';
% c.Label.FontSize = 12;


%%%%%%%%%%%% pc

modelname=['pc' num2str(model)];
pcn=eval(modelname); 

figure;
plot(pcn,'o-','markerfacecolor','black','linewidth',1,'color','black');
cmax=max(max(pcn));
cmin=min(min(pcn));
caxisRange = max(abs(cmin), abs(cmax));
ylim([-caxisRange, caxisRange]);
set(gca,'YDir','normal');
xticks(1:6:xend);
xticklabels(2019:2021);
xtickangle(45);
ax=gca;
ax.XAxis.LineWidth=1;
xlabel('year','fontname','Times New Roman','fontsize',16);%绘制子图x标签
title(modelname,'fontname','Times New Roman','fontsize',17');
hold on
x=[1 xend];
xlim([x(1) x(end)]);
y=[0 0];
plot(x,y,'k--','linewidth',0.8);
hold off

%%%pc合图
figure;
bx=tight_subplot(4,1,[0.003 0.003],[0.1 0.1],[0.1 0.05]);
for pic=1:4
    axes(bx(pic));        
    for model=1:4
        modelname=['pc' num2str(model)];
        pcn=eval(modelname);
        if model==pic
            plot(pcn,'linewidth',2,'displayname',modelname);
            text(-5, 0.5, modelname, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 0,...
                'fontname','Times New Roman','fontsize',15);
        else
            plot(pcn,'--','linewidth',1,'displayname',modelname);
        end
        box off;
        grid on;
        cmax=max(max(pcn));
        cmin=min(min(pcn));
        caxisRange = max(abs(cmin), abs(cmax));
        ylim([-caxisRange, caxisRange]);
        set(gca,'YDir','normal');
        if pic==4
            xticks(1:6:18);
            xticklabels(2019:2021);
            xtickangle(45);        
            xlabel('year','fontname','Times New Roman','fontsize',16);%绘制子图x标签
        else
            xticks(1:6:102);
        end
        hold on
    end
    
    x=[1 18];
    xlim([x(1) x(end)]);
    y=[0 0];
    plot(x,y,'k--','linewidth',0.8);
    hold off
end
sgtitle('pcs','fontname','Times New Roman','fontsize',17);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   全时间段
clc;
clear;

monthbegin = 2;
monthend = 120;
xend=monthend-monthbegin+1;


height=nan(dh,1);
for h=1:dh
height(h,1)=110-h;
end

latgrid=-50:latticewid:50;
lata=latgrid';

for i=monthbegin:monthend
all_point_f0(:,:,i)=all_point_f(:,:,i).*(sqrt(cosd(lata)));
end
all_point_f0(all_point_f0==0)=NaN;

isEmptyMatrix = false(1, size(all_point_f0, 3));
for i = 1:size(all_point_f0, 3)
    isEmptyMatrix(i) = isempty(all_point_f0(:,:,i));
end
all_point_f0 = all_point_f0(:,:,~isEmptyMatrix);


[eof_maps,pc,expvar] = eof(all_point_f0(:,:,2:120));
eof1=-eof_maps(:,:,1).*std(pc(1,:));
pc1=-pc(1,:)./std(pc(1,:));
eof2=-eof_maps(:,:,2).*std(pc(2,:));
pc2=-pc(2,:)./std(pc(2,:));

eof3=eof_maps(:,:,3).*std(pc(3,:));
pc3=pc(3,:)./std(pc(3,:));
eof4=-eof_maps(:,:,4).*std(pc(4,:));
pc4=-pc(4,:)./std(pc(4,:));

eof5=-eof_maps(:,:,5).*std(pc(5,:));
pc5=-pc(5,:)./std(pc(5,:));
eof6=-eof_maps(:,:,6).*std(pc(6,:));
pc6=-pc(6,:)./std(pc(6,:));

%%%%%%%%%%%% eof
model=4;

modelname=['eof' num2str(model)];
eofn=eval(modelname); 
figure;
contourf(latgrid,height,eofn',200,'linecolor','none');
set(gca,'YDir','normal');
ylabel('latitude','fontname','Times New Roman','fontsize',16);
xlabel('longitude','fontname','Times New Roman','fontsize',16);
titlename=horzcat(modelname,'  ','(' ,num2str(expvar(model)),'%',')');
title(titlename,'fontsize',17,'fontname','Times New Roman');
colormap(colors);
c=colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6]);
caxis([-1.8e-5, 1.8e-5]);
c.Ticks= (-1.8:0.4:1.8) * 1e-5; 
% c.TickLabels=arrayfun(@(x) sprintf('%.0e', x), ticks, 'UniformOutput', false);
% c.Label.String = 'Intensity';
% c.Label.FontSize = 12;


%%%%%%%%%%%% pc

modelname=['pc' num2str(model)];
pcn=eval(modelname); 

figure;
plot([NaN,pcn],'o-','markerfacecolor','black','linewidth',1,'color','black');
cmax=max(max(pcn));
cmin=min(min(pcn));
caxisRange = max(abs(cmin), abs(cmax));
ylim([-caxisRange, caxisRange]);
set(gca,'YDir','normal');
xticks(1:6:xend);
xticklabels(2002:2021);
xtickangle(45);
ax=gca;
ax.XAxis.LineWidth=1;
xlabel('year','fontname','Times New Roman','fontsize',16);%绘制子图x标签
title(modelname,'fontname','Times New Roman','fontsize',17');
hold on
x=[1 xend];
xlim([x(1) x(end)]);
y=[0 0];
plot(x,y,'k--','linewidth',0.8);
hold off


%%%pc合图
figure;
bx=tight_subplot(4,1,[0.003 0.003],[0.1 0.1],[0.1 0.05]);
for pic=1:4
    axes(bx(pic));        
    for model=1:4
        modelname=['pc' num2str(model)];
        pcn=eval(modelname);
        if model==pic
            plot([NaN,pcn],'linewidth',2,'displayname',modelname);
            text(-5, 0.5, modelname, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 0,...
                'fontname','Times New Roman','fontsize',15);
        else
            plot([NaN,pcn],'--','linewidth',1,'displayname',modelname);
        end
        box off;
        grid on;
        cmax=max(max(pcn));
        cmin=min(min(pcn));
        caxisRange = max(abs(cmin), abs(cmax));
        ylim([-caxisRange, caxisRange]);
        set(gca,'YDir','normal');
        if pic==4
            xticks(1:6:120);
            xticklabels(2002:2021);
            xtickangle(45);        
            xlabel('year','fontname','Times New Roman','fontsize',16);%绘制子图x标签
        else
            xticks(1:6:120);
        end
        hold on
    end
    
    x=[1 120];
    xlim([x(1) x(end)]);
    y=[0 0];
    plot(x,y,'k--','linewidth',0.8);
    hold off
end
sgtitle('pcs','fontname','Times New Roman','fontsize',17);
 
%% EOF投影
%%%%   02-18年

monthbegin = 2;
monthend = 102;
xend=monthend-monthbegin+1;


height=nan(dh,1);
for h=1:dh
height(h,1)=110-h;
end

latgrid=-50:latticewid:50;
lata=latgrid';

for i=monthbegin:monthend
all_point_f0(:,:,i)=all_point_f(:,:,i).*(sqrt(cosd(lata)));
end
all_point_f0(all_point_f0==0)=NaN;

isEmptyMatrix = false(1, size(all_point_f0, 3));
for i = 1:size(all_point_f0, 3)
    isEmptyMatrix(i) = isempty(all_point_f0(:,:,i));
end
all_point_f0 = all_point_f0(:,:,~isEmptyMatrix);


[eof_maps,pc,expvar] = eof(all_point_f0(:,:,2:102)); %#ok<*ASGLU>
eof1=-eof_maps(:,:,1).*std(pc(1,:));
pc1=-pc(1,:)./std(pc(1,:));
eof2=-eof_maps(:,:,2).*std(pc(2,:));
pc2=-pc(2,:)./std(pc(2,:));

eof3=-eof_maps(:,:,3).*std(pc(3,:));
pc3=-pc(3,:)./std(pc(3,:));
eof4=-eof_maps(:,:,4).*std(pc(4,:));
pc4=-pc(4,:)./std(pc(4,:));


%%%      19-21年
monthbegin=103;
monthend=120;
xend=monthend-monthbegin+1;

height=nan(dh,1);
for h=1:dh
height(h,1)=110-h;
end

latgrid=-50:latticewid:50;
lata=latgrid';

all_point_f0=NaN(size(all_point_f));

for i=monthbegin:monthend
all_point_f0(:,:,i)=all_point_f(:,:,i).*(sqrt(cosd(lata)));
end
all_point_f0(all_point_f0==0)=NaN;

isEmptyMatrix = false(1, size(all_point_f0, 3));
for i = 1:size(all_point_f0, 3)
    isEmptyMatrix(i) = all(all(isnan(all_point_f0(:,:,i))));
end
all_point_f0 = all_point_f0(:,:,~isEmptyMatrix);


% 获取 3 年数据集的时间步数
[~,~,n] = size(all_point_f0);

% 初始化存储投影系数的矩阵
projection_coeffs_3_years = NaN(n, 1);

eofs_17_years_mode1_vector = eof1(:)';
% 对于每个时间步
for i = 1:n
    % 将 3 年数据集的空间场（每个时间步的空间分布）reshape 为列向量
    spatial_pattern = reshape(all_point_f0(:,:,i), [], 1);
    
    % 计算投影系数（即空间场与 17 年 EOF 模态 1 的点积）
    projection_coeffs_3_years(i) = dot(spatial_pattern',eofs_17_years_mode1_vector);
end

% 可选：归一化投影系数（可选）
projection_coeffs_3_years = projection_coeffs_3_years ./ max(abs(projection_coeffs_3_years));

% 可视化投影系数
figure;
plot(projection_coeffs_3_years);
xlabel('时间步');
ylabel('模态投影系数');
title('3 年数据集在 17 年 EOF 模态 1 空间中的模态投影系数');




%% 季节分布

clc;
clear;
load('Trenddata.mat');
load('F107data.mat');
load('QBO30data.mat');
load('QBO10data.mat');
load('ENSOdata.mat');

Nspring_months=[];
for n=1:20
    Nspring_months(1,2*n-1) = 6*n-4;
    Nspring_months(1,2*n)=6*n-3;
end
Nsummer_months=[];
Nsummer_months = [4:6:118];
Nautumn_months = [];
for n=1:20
    Nautumn_months(1,2*n-1) = 6*n-1;
    Nautumn_months(1,2*n)=6*n;
end
Nwinter_months=[];
Nwinter_months = [1:6:115];

Nspring_c=all_point_f(:,:,Nspring_months);
Nsummer_c=all_point_f(:,:,Nsummer_months);
Nautumn_c=all_point_f(:,:,Nautumn_months);
Nwinter_c=all_point_f(:,:,Nwinter_months);

%分区间
%02-21年
syear='2002-2021';
Nspring_ck=Nspring_c;
Nsummer_ck=Nsummer_c;
Nautumn_ck=Nautumn_c;
Nwinter_ck=Nwinter_c;
%02-17年
syear='2002-2017';
Nspring_ck=Nspring_c(:,:,1:32);
Nsummer_ck=Nsummer_c(:,:,1:16);
Nautumn_ck=Nautumn_c(:,:,1:32);
Nwinter_ck=Nwinter_c(:,:,1:16);
%18-21年
syear='2018-2021';
Nspring_ck=Nspring_c(:,:,33:40);
Nsummer_ck=Nsummer_c(:,:,17:20);
Nautumn_ck=Nautumn_c(:,:,33:40);
Nwinter_ck=Nwinter_c(:,:,17:20);

%%%%colormap的设置
colors=[109 001 031;183 034 048;220 109 087;246 178 147;251 227 213;
    233 241 244;182 215 232;109 173 209;049 124 183;016 070 128;];
colors=colors/255;
originalNumColors = size(colors, 1);
newColors = 40;
step = (originalNumColors - 1) / (newColors - 1);
interpIndices = 1:step:originalNumColors;
interpColors = interp1(1:originalNumColors, colors, interpIndices);
colors=flipud(interpColors);

height=nan(dh,1);
for h=1:dh
height(h,1)=110-h;
end
latgrid=-54:latticewid:54;

figure;
h=tight_subplot(2,2,[0.03 0.04],[0.1 0.1],[0.06 0.1]);
for s=1:4
    axes(h(s));
    season={'spring';'summer';'autumn';'winter'};
    season=cell2mat(season(s));
    seasonckname=['N' ,season,'_ck'];
    Nseason_ck=eval(seasonckname);
    
    season_mean = nan(size(Nseason_ck, 1), size(Nseason_ck, 2));
    season_mean=nanmean(Nseason_ck,3)
    
    contourf(latgrid,height,season_mean',40,'linecolor','none');
    set(gca,'YDir','normal');
    ylabel('altitude','fontname','Times New Roman','fontsize',13);
    if (s==1||s==2)
        xticks(-50:10:50)
        xticklabels([]);
    end
    if (s==3||s==4)
        xlabel('latitude','fontname','Times New Roman','fontsize',13);
    end
    box off;
    grid on;
    titlename=['N',season];
    title(titlename,'fontsize',13,'fontname','Times New Roman');
    % cmax=max(max(Nspring_data));
    % cmin=min(min(Nspring_data));
    % caxisRange = max(abs(cmin), abs(cmax));
    caxis([0.00008,0.0004]);
    colormap(colors);
    colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6]);
    hold on;
end
sgtitle([syear,'   65-110km CO2VMR Distribution'],'fontname','Times New Roman','fontsize',17);


%%%对all_point_season做eof(月份单位数据，用上面的季节代码换到季节数据)
clc;clear;
%%%%colormap的设置
colors=[109 001 031;183 034 048;220 109 087;246 178 147;251 227 213;
    233 241 244;182 215 232;109 173 209;049 124 183;016 070 128;];
colors=colors/255;
originalNumColors = size(colors, 1);
newColors = 40;
step = (originalNumColors - 1) / (newColors - 1);
interpIndices = 1:step:originalNumColors;
interpColors = interp1(1:originalNumColors, colors, interpIndices);
colors=flipud(interpColors);


height=nan(dh,1);
for h=1:dh
height(h,1)=140-h;
end

latgrid=-90:latticewid:90;
lata=latgrid';
all_point_season0=all_point_season.*(sqrt(cosd(lata)));
[eof_maps,pc,expvar] = eof(all_point_season0);

eof1=-eof_maps(:,:,1).*std(pc(1,:));
pc1=-pc(1,:)./std(pc(1,:));
eof2=-eof_maps(:,:,2).*std(pc(2,:));
pc2=-pc(2,:)./std(pc(2,:));

eof3=-eof_maps(:,:,3).*std(pc(3,:));
pc3=-pc(3,:)./std(pc(3,:));
eof4=-eof_maps(:,:,4).*std(pc(4,:));
pc4=-pc(4,:)./std(pc(4,:));

eof5=-eof_maps(:,:,5).*std(pc(5,:));
pc5=-pc(5,:)./std(pc(5,:));
eof6=-eof_maps(:,:,6).*std(pc(6,:));
pc6=-pc(6,:)./std(pc(6,:));

%%%%%%%%%%%% eof
model=2;

modelname=['eof' num2str(model)];
eofn=eval(modelname); 
figure;
contourf(latgrid,height,eofn',200,'linecolor','none');
set(gca,'YDir','normal');
ylabel('latitude','fontname','Times New Roman','fontsize',16);
xlabel('longitude','fontname','Times New Roman','fontsize',16);
%%'position',[185,-90,0]
titlename=horzcat(modelname,'  ','(' ,num2str(expvar(model)),'%',')');
title(titlename,'fontsize',17,'fontname','Times New Roman');
colormap(colors);
colorbar('fontname','Times New Roman','fontsize',15,'Position',[0.92,0.22,0.015,0.6]);


%%%%%%%%%%%% pc
model=3;

modelname=['pc' num2str(model)];
pcn=eval(modelname); 
figure;

plot(pcn,'o-','markerfacecolor','black','linewidth',...
        1,'color','black');
cmax=max(max(pcn));
cmin=min(min(pcn));
caxisRange = max(abs(cmin), abs(cmax));
ylim([-caxisRange, caxisRange]);
set(gca,'YDir','normal');


xtickangle(45);
ax=gca;
ax.XAxis.LineWidth=1;
xlabel('year','fontname','Times New Roman','fontsize',16);%绘制子图x标签
title(modelname,'fontname','Times New Roman','fontsize',17');
hold on
x=[1 79];
xlim([x(1) x(end)]);
y=[0 0];
plot(x,y,'k--','linewidth',0.8);
hold off



%}

xpbombs;



