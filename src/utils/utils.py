# coding:utf-8

import pickle


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def load_file(fp: str, sep: str = None, name_tuple=None):
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            if name_tuple:
                return map(name_tuple._make, [line.strip().split(sep) for line in lines])
            else:
                return [line.strip().split(sep) for line in lines]
        else:
            return lines


def load_pkl(fp):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, fp):
    with open(fp, 'wb') as f:
        pickle.dump(data, f)
