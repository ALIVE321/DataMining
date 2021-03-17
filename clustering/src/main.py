from utils import *


if __name__ == "__main__":
    data = read_data()
    res = cluster(data)
    # print(res)
    write_res(res)
