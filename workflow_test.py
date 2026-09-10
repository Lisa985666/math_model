from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 输出路径以本代码所在位置为准
output_dir = Path(__file__).resolve().parent / "workflow_test_output"
output_dir.mkdir(exist_ok=True)

print("当前解释器：", sys.executable)

# 1. 生成模拟数据并保存为 Excel
input_file = output_dir / "sample_data.xlsx"
sample = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6],
    "y": [3.1, 4.9, 7.2, 8.8, 11.1, 12.9]
})
sample.to_excel(input_file, index=False)

# 2. 从 Excel 读取数据
data = pd.read_excel(input_file)
x = data["x"].to_numpy()
y = data["y"].to_numpy()

# 3. 拟合直线 y = kx + b
k, b = np.polyfit(x, y, 1)
prediction = k * x + b
rmse = np.sqrt(np.mean((y - prediction) ** 2))

print(f"拟合结果：y = {k:.4f}x + {b:.4f}")
print(f"均方根误差 RMSE：{rmse:.4f}")

# 4. 保存计算结果
data["prediction"] = prediction
data["residual"] = y - prediction
data.to_excel(output_dir / "fit_results.xlsx", index=False)

# 5. 绘图，测试中文显示和图片导出
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x, y, label="原始数据", color="#2878B5")
ax.plot(x, prediction, label="拟合直线", color="#E07B39")
ax.set_title("数模环境测试：线性拟合")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()

fig.savefig(output_dir / "fit_plot.png", dpi=300)
fig.savefig(output_dir / "fit_plot.pdf")

print("\n完整流程运行成功！")
print("文件保存位置：", output_dir)

plt.show()