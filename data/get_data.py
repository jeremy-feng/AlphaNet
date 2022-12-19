from rich.progress import Progress
from datetime import timedelta 
import pandas as pd
import numpy as np
import tushare as ts
pro = ts.pro_api()


# 给定一个交易日，返回该日满足条件的A股股票列表
def get_stocklist(date: str, num: int):

    start = str(pd.to_datetime(date)-timedelta(30))
    start = start[0:4]+start[5:7]+start[8:10]
    df1 = pro.index_weight(index_code='000002.SH',
                           start_date=start, end_date=date)  # 交易日当天的股票列表
    codes = list(df1['con_code'])
    codes = codes[0:1000]  # 在每个截面期只选取1000只股票

    return codes

# 给定日期区间的端点，输出期间的定长采样交易日列表


def get_datelist(start: str, end: str, interval: int):

    df = pro.index_daily(ts_code='399300.SZ', start_date=start, end_date=end)
    date_list = list(df.iloc[::-1]['trade_date'])
    sample_list = []
    for i in range(len(date_list)):
        if i % interval == 0:
            sample_list.append(date_list[i])

    return sample_list

# 返回两个，一个是前30个交易日的9个指标面板（9*30），一个是未来10天的收益率


def get_x_y(code: str, date: str, pass_day: int, future_day: int, len1: int, len2: int):

    start = str(pd.to_datetime(date)-timedelta(pass_day*2))
    start = start[0:4]+start[5:7]+start[8:10]
    end = str(pd.to_datetime(date)+timedelta(future_day*2))
    end = end[0:4]+end[5:7]+end[8:10]
    df_price = pro.daily(ts_code=code,  # OHLC,pct_change,volume
                         start_date=start, end_date=date)
    df_basic = pro.daily_basic(ts_code=code,
                               start_date=start, end_date=date)
    df_return = pro.daily(ts_code=code,
                          start_date=date, end_date=end).iloc[::-1]['close']
    if (df_price.shape[0] == df_basic.shape[0]) & (df_price.shape[0] == len1) & (df_return.shape[0] == len2):  # 判断数据的完整性
        df_price = df_price.iloc[0:pass_day, [2, 3, 4, 5, 8, 9]].fillna(0.1)
        df_basic = df_basic.iloc[0:pass_day, [3, 4, 5]].fillna(0.1)
        data = np.array(pd.merge(df_price, df_basic,
                        left_index=True, right_index=True).iloc[::-1].T)
        # print(data.shape)
        # 未来十个交易日的收益率
        dfr = df_return.iloc[0:future_day]
        ret = dfr.iloc[-1]/dfr.iloc[0]-1  # 后十个交易日的收益率
        return data, ret
    else:
        return None, None  # 数据缺失的预处理


def get_length(date: str, pass_day: int, future_day: int):
    start = str(pd.to_datetime(date)-timedelta(pass_day*2))
    start = start[0:4]+start[5:7]+start[8:10]
    end = str(pd.to_datetime(date)+timedelta(future_day*2))
    end = end[0:4]+end[5:7]+end[8:10]
    len_1 = pro.index_daily(ts_code='399300.SZ',
                            start_date=start, end_date=date).shape[0]
    len_2 = pro.index_daily(ts_code='399300.SZ',
                            start_date=date, end_date=end).shape[0]
    return len_1, len_2

# 构造数据集的函数：输入一个时间区间的端点，得到该区间内采样交易日期的所有数据


def get_dataset(num: int, start: str, end: str, interval: int, pass_day: int, future_day: int):
    X_train = []
    y_train = []
    trade_date_list = get_datelist(start, end, interval)
    # 添加进度条
    with Progress() as progress:
        task_date = progress.add_task(
            "[red]Date...", total=len(trade_date_list))
        for date in trade_date_list:
            # 更新进度条
            progress.update(task_date, advance=1)
            stock_list = get_stocklist(date, num)
            len1, len2 = get_length(date, pass_day, future_day)
            task_stock = progress.add_task(
                "[green]Stock...", total=len(range(len(stock_list))))
            for i in range(len(stock_list)):
                # 更新进度条
                progress.update(task_stock, advance=1)
                code = stock_list[i]
                x, y = get_x_y(code, date, pass_day, future_day, len1, len2)
                try:
                    if (x.shape[0] == 9) & (x.shape[1] == pass_day):
                        X_train.append(x)
                        y_train.append(y)
                except Exception:
                    continue
    return X_train, y_train


# 参数设定：使用过去30天的数据预测未来10天的收益率，回归问题
X_train, y_train = get_dataset(
    num=1000, start='20220101', end='20220630', interval=10, pass_day=30, future_day=10)
X_test, y_test = get_dataset(num=1000, start='20220931',
                             end='20221231', interval=10, pass_day=30, future_day=10)
print("there are in total", len(X_train), "training samples")
print("there are in total", len(X_test), "testing samples")
# 将数据保存到本地供离线训练
Xa = np.array(X_train)
ya = np.array(y_train)
Xe = np.array(X_test)
ye = np.array(y_test)
np.save('./X_train.npy', Xa)
np.save('./y_train.npy', ya)
np.save('./X_test.npy', Xe)
np.save('./y_test.npy', ye)
