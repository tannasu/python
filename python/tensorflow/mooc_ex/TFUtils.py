import csv
import scipy.io as sio
import numpy as np

def open_csv(path):
    with open(path) as csv_file:
        ex_data = list(csv.reader(csv_file))
    return ex_data

def load_mat(path):
    mat_files = sio.loadmat(path)
    y_data = mat_files["y"]
    X_data = mat_files["X"]
    return y_data, X_data

def create_data(csv_data, is_X=True):
    data_list = []
    sub_length = len(csv_data[0])
    if is_X:
        from_a = 0
        from_b = sub_length - 1
    else:
        from_a = sub_length - 1
        from_b = sub_length
    for sub_data in csv_data:
        data_list.append(sub_data[from_a:from_b])
    return data_list


def stack_img_data(data_list, single_weight, img_block_size):
    img_data = None
    for x in range(img_block_size):
        column_stack = None
        for y in range(img_block_size):
            img_index = x * img_block_size + y
            single_data = data_list[img_index]
            #不转置图像是横着的……
            single_data = single_data.reshape(single_weight, single_weight).T
            if column_stack is None:
                column_stack = single_data
            else:
                column_stack = np.column_stack((column_stack, single_data))
        if img_data is None:
            img_data = column_stack
        else:
            img_data = np.row_stack((img_data, column_stack))
    return img_data

#load_mat("data/ex3data1.mat")