# 第五章 实验设计与结果分析

## 5.1 实验环境

本文基于 Python、Flask、PyTorch、Scikit-learn、Pandas 等工具实现股票价格预测与交易回测实验。系统通过 Tushare 获取股票历史行情数据，并在统一的数据预处理、训练集/测试集划分和评价指标下开展模型对比实验与特征消融实验。

## 5.2 数据集说明

实验数据包括个股历史交易数据和大盘指数数据。个股数据主要包含开盘价、最高价、最低价、收盘价、成交额等基础行情字段；在此基础上进一步构造 MA5、MA10、MACD、RSI、VOLATILITY 等技术指标，以及 RET、LOG_RET、INDEX_RET 等收益率特征。所有特征在训练前进行归一化处理，并按照时间顺序划分训练集和测试集。

## 5.3 评价指标

为全面评估模型效果，本文同时使用价格预测指标、方向预测指标和回测评价指标。

价格预测指标包括 MSE、RMSE、MAE 和 MAPE，用于衡量预测价格与真实价格之间的误差。

方向预测指标包括 Accuracy、Precision、Recall 和 F1-score，用于衡量模型对股票涨跌方向的判断能力。

回测评价指标包括 Total Return、Max Drawdown、Sharpe Ratio 和 Trade Cycles，用于衡量交易策略在测试区间内的收益、风险和交易行为。

## 5.4 模型对比实验

为验证本文模型的有效性，本文选取 Naive Baseline、Linear Regression、Random Forest、Vanilla LSTM 和 Attention-LSTM 作为对比模型，在相同训练集、测试集和评价指标下进行实验。

Naive Baseline 假设下一交易日收盘价等于当前交易日收盘价；Linear Regression 和 Random Forest 作为传统机器学习方法；Vanilla LSTM 作为不带注意力机制的循环神经网络模型；Attention-LSTM 则在 LSTM 基础上引入注意力机制，用于增强模型对关键时间步信息的捕捉能力。

## 5.5 特征消融实验

为分析不同特征对预测结果的影响，本文设计了特征消融实验。实验组包括 Full Features、No Technical Indicators、No Market Index、No Return Features 和 Only Price Features。

Full Features 使用全部特征；No Technical Indicators 去除 MA5、MA10、MACD、RSI、VOLATILITY 等技术指标；No Market Index 去除大盘指数及相关收益率特征；No Return Features 去除 RET、LOG_RET、INDEX_RET；Only Price Features 仅保留开盘价、最高价、最低价、收盘价和成交额。

通过比较不同实验组的预测误差，可以分析技术指标、大盘特征和收益率特征对模型性能的贡献。

## 5.6 回测结果分析

在获得预测结果后，系统根据预测方向生成交易信号，并结合买入阈值、卖出阈值和交易费用进行策略回测。回测结果通过总收益率、最大回撤、夏普比率和交易轮数进行评估。

模型对比实验与特征消融实验的结果将分别导出为 `static/results/model_comparison.csv` 和 `static/results/ablation_result.csv`，可直接用于论文表格、结果分析和后续可视化。
