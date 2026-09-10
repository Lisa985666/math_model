import sys
import statistics


def main():
    print("=== 数模项目运行测试 ===")
    print("Python 版本：", sys.version.split()[0])
    print("解释器路径：", sys.executable)

    # 模拟一组实验数据
    data = [12, 15, 18, 21, 24]

    print("\n实验数据：", data)
    print("数据总和：", sum(data))
    print("平均值：", statistics.mean(data))
    print("最大值：", max(data))
    print("最小值：", min(data))

    # 检查计算结果
    assert sum(data) == 90, "数据总和计算异常"
    assert statistics.mean(data) == 18, "平均值计算异常"

    print("\n测试通过！Python 可以正常运行。")
    print("这是准备上传到 GitHub 的第一个版本。")


if __name__ == "__main__":
    main()