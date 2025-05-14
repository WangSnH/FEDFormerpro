import numpy as np
import pandas as pd
import argparse
def main():
    parser = argparse.ArgumentParser(description="csv")
    parser.add_argument('--file_path', type=str, default='results/test/real_prediction.npy', help='file_path')
    args = parser.parse_args()
    file_path = args.file_path# 替换为你的文件路径
    data = np.load(file_path)

    # 打印完整数据内容
    print("数据内容：")
    print(data)

    # 取第一行作为列名
    new_header = data[0]

    # 从第二行开始加载数据
    data_2d = data[1:]

    # 转换为DataFrame，并设置第一行为列名
    df = pd.DataFrame(data_2d, columns=new_header)

    # 生成CSV保存路径（原路径同级目录）
    csv_path = file_path.replace('.npy', '.csv')

    # 保存为CSV，不添加新的索引
    df.to_csv(csv_path, index=False)

    print(f"转换完成！CSV文件已保存至：{csv_path}")
if __name__ == "__main__":
    main()