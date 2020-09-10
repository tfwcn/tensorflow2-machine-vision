import os
import re

def ReadFileList(dir_path, pattern=r'.*', select_sub_path=True):
    '''读取文件列表'''
    # 文件列表
    files = []
    # 正则
    pattern = re.compile(pattern)
    # 查询子路径
    if select_sub_path==1:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for f in filenames:
                # 转小写
                file_name = f.lower()
                file_path=os.path.join(dirpath, f)
                match = pattern.search(file_name)
                if match:
                    files.append(file_path)
    else:
        # 只查询当前目录文件
        for file_or_dir_name in os.listdir(dir_path):
            if os.path.isfile(file_or_dir_name):
                f = file_or_dir_name
                # 转小写
                file_name = f.lower()
                file_path=os.path.join(dir_path, f)
                match = pattern.search(file_name)
                if match:
                    files.append(file_path)
    return files


def ReadPathList(dir_path, pattern=r'.*', select_sub_path=True):
    '''读取文件夹列表'''
    # 文件列表
    files = []
    # 正则
    pattern = re.compile(pattern)
    # 查询子路径
    if select_sub_path==1:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for f in dirnames:
                # 转小写
                file_name = f.lower()
                file_path=os.path.join(dirpath, f)
                match = pattern.search(file_name)
                if match:
                    files.append(file_path)
    else:
        # 只查询当前目录文件
        for file_or_dir_name in os.listdir(dir_path):
            if os.path.isdir(file_or_dir_name):
                f = file_or_dir_name
                # 转小写
                file_name = f.lower()
                file_path=os.path.join(dir_path, f)
                match = pattern.search(file_name)
                if match:
                    files.append(file_path)
    return files
