# 第五章 实验设计与结果分析

## 5.1 实验环境

本文基于 Python、Flask、PyTorch、Scikit-learn、Pandas 等工具实现股票收盘价预测实验。系统通过 Tushare 获取股票历史行情数据，并在统一的数据预处理、训练集/测试集划分和评价指标下开展模型对比实验与特征消融实验。

## 5.2 数据集说明

实验数据包括个股历史交易数据和大盘指数数据。个股数据主要包含开盘价、最高价、最低价、收盘价、成交额等基础行情字段；在此基础上进一步构造 MA5、MA10、MACD、RSI、VOLATILITY 等技术指标，以及 RET、LOG_RET、INDEX_RET 等收益率特征。所有特征在训练前进行归一化处理，并按照时间顺序划分训练集和测试集。

## 5.3 评价指标

为全面评估模型效果，本文使用价格预测指标衡量预测收盘价与真实收盘价之间的误差。

价格预测指标包括 RMSE、MAE、MAPE 和 R²。其中 RMSE、MAE 和 MAPE 用于衡量预测价格与真实价格之间的误差，R² 用于衡量模型对真实收盘价波动的解释能力。

涨跌概率信号指标包括方向命中率、平均上涨概率、模型看涨比例和买入信号命中率，用于评估 Enhanced Vanilla LSTM 输出的交易概率是否具备信号价值。

回测评价指标包括 Total Return、Max Drawdown、Sharpe Ratio、期末资产、交易次数、胜率、平均仓位和累计交易成本。系统基于模型输出的上涨概率、手续费、最低佣金、卖出税费、滑点、初始资金、整手交易、T+1 约束和仓位约束执行日线策略回测，并同时展示策略累计收益和买入持有收益。

## 5.4 模型对比实验

为验证本文模型的有效性，本文选取 ARIMA、DNN、CNN 和 Vanilla LSTM 作为对比模型，在相同训练集、测试集和评价指标下进行实验。

ARIMA 作为传统时间序列模型；DNN 和 CNN 作为深度学习对比模型；Vanilla LSTM 作为循环神经网络模型，用于建模股票时间序列中的长期依赖关系。

## 5.5 特征消融实验

为分析不同特征对预测结果的影响，本文设计了特征消融实验。实验组包括 Full Features、No Technical Indicators、No Market Index、No Return Features 和 Only Price Features。

Full Features 使用全部特征；No Technical Indicators 去除 MA5、MA10、MACD、RSI、VOLATILITY 等技术指标；No Market Index 去除大盘指数及相关收益率特征；No Return Features 去除 RET、LOG_RET、INDEX_RET；Only Price Features 仅保留开盘价、最高价、最低价、收盘价和成交额。

通过比较不同实验组的预测误差，可以分析技术指标、大盘特征和收益率特征对模型性能的贡献。

## 5.6 实验结果分析

在获得预测结果后，系统将不同模型和不同特征组合下的 RMSE、MAE、MAPE 和 R² 进行对比，从而分析模型结构与特征类型对收盘价预测性能的影响。同时，系统结合 Enhanced Vanilla LSTM 涨跌概率信号进行日线策略回测，对比策略累计收益与买入持有收益。

模型对比实验与特征消融实验的结果将分别导出为 `static/results/model_comparison.csv` 和 `static/results/ablation_result.csv`，可直接用于论文表格、结果分析和后续可视化。
