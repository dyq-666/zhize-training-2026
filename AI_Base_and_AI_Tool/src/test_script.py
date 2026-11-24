import os
import sys

def main():
    print("=== 智泽实验室 A.1 任务测试脚本 ===")
    print(f"当前使用的 Python 版本: {sys.version.split()[0]}")
    print(f"当前 Linux 用户: {os.environ.get('USER')}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 简单的数据处理
    numbers = [1, 2, 3, 4, 5]
    squared = [n**2 for n in numbers]
    print(f"测试数据处理：{numbers} 的平方是 {squared}")
    print("环境配置成功！")

if __name__ == "__main__":
    main()
