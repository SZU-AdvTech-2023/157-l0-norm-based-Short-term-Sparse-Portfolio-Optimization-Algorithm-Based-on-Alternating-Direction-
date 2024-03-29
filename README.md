# Introduce 介绍

论文提出了一种新的基于0-范数的短期稀疏投资组合优化模型。与现有方法相比，该模型根据资产的短期增长潜力选择投资组合，并引入0-范数约束直接控制所选投资组合中非零资产的最大数量。与基于1-范数的方法不同，本文的模型中可以直接使用禁止卖空约束。此外，引入稀疏正则化项以消除SSPO系统中的琐碎交易。此外，为了求解包含的非凸优化系统，提出了一种基于交替方向乘子法（ADMM）概念的算法。文中还研究了算法的收敛性。最后，通过在四个真实数据集上的数值实验证明了该方法的有效性。

# Structure 结构

项目结构如下：

- **algos** 论文算法以及对比的算法
- **data** 数据集
- **tools** 公用工具
- **result** 结果处理工具
- **algos** 算法工具


## Usage 用法

调用tools.quickrun()方法，指定运行的算法以及数据集，返回算法运行结果。


