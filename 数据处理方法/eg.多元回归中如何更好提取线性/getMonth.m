

function [month, half] = getMonth(year, day)
    leapYear = isLeapYear(year); % 判断是否为闰年
    daysInMonth = [31 28+leapYear 31 30 31 30 31 31 30 31 30 31]; % 每个月的天数
    cumulativeDays = cumsum(daysInMonth); % 计算每个月累计的天数
    month = find(day <= cumulativeDays, 1); % 通过查找累计天数，确定月份
    
    % 判断上半月还是下半月
    if (month==1)
        if day <= 15
            half = 1;
        else
            half = 2;
        end
    else
        if (day-cumulativeDays(month-1)<=15)
            half = 1;
        else
            half = 2;
        end
    end
end

function leapYear = isLeapYear(year)
    leapYear = mod(year, 4) == 0 && (mod(year, 100) ~= 0 || mod(year, 400) == 0); % 判断是否为闰年
end