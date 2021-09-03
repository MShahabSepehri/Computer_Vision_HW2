import configparser
import os


def get_config(path, key):
    parent_dir = os.getcwd()
    config = configparser.ConfigParser()
    config.read(parent_dir + '/Config/' + path)
    params_parser = config[key]
    dic = {}
    for key in params_parser:
        try:
            # noinspection PyTypeChecker
            dic[key] = float(params_parser[key])
        except:
            dic[key] = params_parser[key]
    return dic
