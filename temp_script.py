#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-08 16:57
# @Author  : Joe
# @Site    : 
# @File    : temp_script.py
# @Software: PyCharm



dataset_sizes = {'sample': (40, 10), 'small': (1280, 320), 'medium': (32000, 8000), 'large': (2000000, 400000)}
num_train, num_test = dataset_sizes["small"]
print(num_train, num_test)