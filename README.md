# Insurance Conversion Modeling

这是我在课程中做的保险客户成交预测项目。我用它练习从数据处理、分箱和特征筛选到分类建模的完整流程，也关注模型换到后续月份后是否仍然稳定。

仓库保留课程代码，并提供一个合成数据示例，方便在不接触客户数据的情况下运行分箱和预测流程。

## 代码里做了什么

我把课程中的主要流程放在 `optimized_homework.py`：监督分箱、IV及相关性筛选、逻辑回归与梯度提升树、前推验证、逐月指标、PSI、Bootstrap及业务容量分析。代码包含中文注释。

客户数据、逐客户预测和原始报告没有上传。完整实验需要有权使用的课程数据。

## 先运行一个示例

```bash
python -m pip install -r requirements-demo.txt
python synthetic_demo.py --output-dir outputs/synthetic
```

该示例生成合成数据，复用课程代码中的 `WoeBinner`，在训练期拟合分箱并训练逻辑回归，在之后的样本上输出 AUC、Brier score、样本数和分箱数。这些输出只用于观察合成数据上的模型行为；原课程的完整特征选优流程需要单独运行。

## 授权数据的完整实验

```bash
python optimized_homework.py --data-dir /path/to/authorized-data --output-dir outputs/full
```

输入文件名与字典结构见脚本中的 `CSV_NAME`、`DICT_NAME` 及读取逻辑。原实现包含固定样本数、特征数等课程数据断言，不能直接对任意数据套用。

`verify_optimized.py` 用来核对课程结果与 Excel 文件，需要完整数据和对应输出。原环境中的 Excel 导出工具没有随仓库提供。

## 结果应该怎样理解

- 训练分箱和参数选择不得使用外部测试标签。
- 已经查看过的验证集不能再次称为全新盲测。
- 样本条件概率不等于真实业务总体成交率；未提供成本和收益时不推演实际ROI。
- 原数据的字段生成时点与标签窗口尚需业务核实，不能宣称已排除全部时间泄漏。

我使用 AI 辅助开发和代码整理。合成示例的运行结果、依赖环境，以及完整课程实验尚未重跑的部分，记录在 [运行记录](docs/VALIDATION.md)。
