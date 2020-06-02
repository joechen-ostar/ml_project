#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joe
# @Site    : 
# @File    : ipt.py
# @Software: PyCharm

import ip2Region

def ip2city(ip):
    dbFile = "../data/ip2region.db"
    # algorithm = "momery"
    searcher = ip2Region.Ip2Region(dbFile)
    try:
        data = searcher.memorySearch(ip)
        print(data["region"].decode("utf-8"))
        return data["region"].decode("utf-8").split("|")[-2]
    finally:
        return "0"

ip2city("222.70.168.243")
