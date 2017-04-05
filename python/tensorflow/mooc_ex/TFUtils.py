import csv


def open_csv(path):
    with open(path) as csv_file:
        ex_data = list(csv.reader(csv_file))
    return ex_data

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

