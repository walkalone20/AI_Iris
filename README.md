# AI_Iris

本项目为初学机器学期而写的入门级监督学习项目——根据鸢尾花的四个参数决定它所属的种类。具体学习数据位于 Train 文件下下的 Iris.csv，仅 150 条数据。具体代码位于 main.py。

首先将数据进行归一化，将三种鸢尾花种类使用数字 $1,2,3$ 进行编号，使用如下线性公式拟合：

$$
y_{i}=\sum_{i=0}^{3} w_{j} x_{i,j}+b=\vec{w}\cdot \vec{x_i}+b
$$

然后采用使用梯度下降和正则化，使用如下公式进行更新：

$$
\begin{aligned}
w_j &\leftarrow \beta w_j - \dfrac{\alpha}{m} \sum_{i=1}^m (\vec{w} \cdot \vec{x_i}+b-y_i)x_{i,j}\\
b &\leftarrow \beta w_j - \dfrac{\alpha}{m} \sum_{i=1}^m (\vec{w} \cdot \vec{x_i}+b-y_i)
\end{aligned}
$$

参数 $\beta=0.99999$，学习率为 $\alpha=0.003$，误差精度 $\epsilon=10^{-8}$。
