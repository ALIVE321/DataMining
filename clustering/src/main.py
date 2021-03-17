from utils import *


if __name__ == "__main__":
    data = read_data()
    res = cluster(data)
    write_res(res)
