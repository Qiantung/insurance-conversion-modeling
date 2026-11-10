# Insurance Conversion Modeling

保险客户成交预测与稳定性分析。课程项目，关注训练期分箱、WOE/IV、模型对比、跨期表现和预测不确定性。

## 项目内容

`optimized_homework.py` 保留原课程实现：监督分箱、IV及相关性筛选、逻辑回归与梯度提升树、前推验证、逐月指标、PSI、Bootstrap及业务容量分析。代码包含中文注释。

公开整理包不包含客户数据、逐客户预测、样本编号、模型文件或原始报告。真实数据实验的完整重跑需要有权使用的课程数据；本仓库不提供下载渠道。

## 无私人数据的可运行示例

```bash
python -m pip install -r requirements-demo.txt
python synthetic_demo.py --output-dir outputs/synthetic
```

该示例生成合成数据，复用课程代码中的 `WoeBinner`，在训练期拟合分箱并训练逻辑回归，在之后的样本上输出 AUC、Brier score、样本数和分箱数。不代表真实客户或原课程成绩，不执行原稿的204特征完整选优过程。

## 授权数据的完整实验

```bash
python optimized_homework.py --data-dir /path/to/authorized-data --output-dir outputs/full
```

输入文件名与字典结构见脚本中的 `CSV_NAME`、`DICT_NAME` 及读取逻辑。原实现包含固定样本数、特征数等课程数据断言，不能直接对任意数据套用。

`verify_optimized.py` 是原课程结果及Excel的核验脚本；需要完整数据和原交付文件，不用于合成示例的验收。特定Codex运行时的Excel导出程序不作为通用运行依赖收录。

## 评价边界

- 训练分箱和参数选择不得使用外部测试标签。
- 已经查看过的验证集不能再次称为全新盲测。
- 样本条件概率不等于真实业务总体成交率；未提供成本和收益时不推演实际ROI。
- 原数据的字段生成时点与标签窗口尚需业务核实，不能宣称已排除全部时间泄漏。

项目采用 AI 辅助整理与开发；真实数据的成果归属、公开权限和个人分工在正式发布前继续核实。验证范围见 [验证记录](docs/VALIDATION.md)。
