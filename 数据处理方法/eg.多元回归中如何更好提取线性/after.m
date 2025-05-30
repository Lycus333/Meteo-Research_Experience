
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

