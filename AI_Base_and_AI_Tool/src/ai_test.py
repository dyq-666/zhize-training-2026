from datetime import datetime


def main():
    # 打印当前系统时间
    now = datetime.now()
    print(f"当前系统时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # 计算 1 到 100 的累加和
    total = 0
    for n in range(1, 101):
        total += n
    print(f"1 到 100 的累加和: {total}")


if __name__ == "__main__":
    main()
