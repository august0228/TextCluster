# -*- coding: utf-8 -*-
import datetime
import multiprocessing
import os
import argparse
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool

import pymysql
from tqdm import tqdm

from utils.similar import jaccard
from utils.segmentor import Segmentor
from utils.utils import check_file, ensure_dir, clean_dir, sample_dict, get_stop_words, line_counter, Range

connection = pymysql.connect(host='47.99.87.74',
                             user='august',
                             password='august',
                             db='august',
                             port=33306
                             )
cursor = connection.cursor()


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='./data/infile', help='Directory of input file.')
    parser.add_argument('--output', type=str, default='./data/output', help='Directory to save output file.')
    parser.add_argument('--dict', type=str, default='./data/seg_dict', help='Directory of dict file.')
    parser.add_argument('--stop_words', type=str, default='./data/stop_words', help='Directory of stop words.')
    parser.add_argument('--sample_number', type=int, default=10, choices=[Range(1)],
                        help='Sample number for each bucket.')
    parser.add_argument('--threshold', type=float, default=0.3, choices=[Range(0.0, 1.0)],
                        help='Threshold for matching.')
    parser.add_argument('--name_len', type=int, default=9, choices=[Range(2)], help='Filename length.')
    parser.add_argument('--name_len_update', type=bool, default=False, help='To update file name length.')
    parser.add_argument('--lang', type=str, choices=['cn', 'en'], default='cn', help='Segmentor language setting.')
    args = parser.parse_args()
    return args


def lstg(num, lst):
    for i in range(0, len(lst), num):
        yield lst[i:i + num]


args = _get_parser()
seg = Segmentor(args)

today = time.strftime("%Y%m%d", time.localtime(time.time()))

# 停用词缓存
stop_words_cache = {}
jieba_cache = {}
# load stop words
stop_words = get_stop_words(args.stop_words) if os.path.exists(args.stop_words) else list()

def fenci(i):
    result = {}
    for zzz in i:
        inline = zzz.rstrip()
        line = inline.split(':::')[0]
        result[line] = list(seg.cut(line))
    return result



def main():
    global connection, cursor
    cpu = multiprocessing.cpu_count()
    print("CPU {}".format(cpu))
    # preliminary work
    check_file(args.infile)
    ensure_dir(args.output)
    all_lines = 0
    if args.name_len_update:
        line_cnt = line_counter(args.infile)
        args.name_len = len(str(line_cnt)) + 1

    clean_dir(args.output, args.name_len)
    # end preliminary work

    all_bucked = defaultdict(list)
    p_bucket = defaultdict(list)
    save_idx = 0
    id_name = '{0:0' + str(args.name_len) + 'd}'

    # load tokenizer

    print('Splitting sentence into different clusters ...')
    infile = open(args.infile, 'r', encoding="utf-8")
    i = 0
    all_data = infile.readlines()
    n = 10000  # 大列表中几个数据组成一个小列表
    lstgs = [all_data[i:i + n] for i in range(0, len(all_data), n)]
    print(len(lstgs))
    r = []
    tr = []
    pool = multiprocessing.Pool(processes=4)
    for xyz in lstgs:
        tr.append(pool.apply_async(fenci, (xyz,)))
    pool.close()
    pool.join()

    for res in tr:
        tmp = res.get()
        for z in tmp:
            if z not in jieba_cache.keys():
                jieba_cache[z] = tmp[z]
            else:
                print(z)
    for st in stop_words:
        stop_words_cache[st] = 1

    r.clear()
    r = None

    all_lines = len(jieba_cache)
    print("开始执行 总 {} 行".format(all_lines))
    print("缓存成功jieba {}".format(len(jieba_cache)))
    print("缓存成功停用词 {}".format(len(stop_words_cache)))
    all_data = jieba_cache.keys()
    for inline in all_data:
        if inline == '太原去贵阳怎么走':
            print("")
        i = i + 1
        print("当前第 {} 行----总 {}".format(i, all_lines))
        inline = inline.rstrip()
        line = inline.split(':::')[0]
        is_match = False
        seg_list = jieba_cache[line]
        llll = []
        if stop_words:
            for mmmm in seg_list:
                if mmmm not in stop_words_cache.keys():
                    llll.append(mmmm)
            seg_list = llll
        for wd in seg_list:
            if is_match:
                break
            w_bucket = p_bucket[wd]
            for bucket in w_bucket:
                array = all_bucked[bucket]
                selected = sample_dict(array, args.sample_number)
                selected = list(map(lambda x: x.split(':::')[0], selected))
                selected = list(map(lambda x: jieba_cache[x], selected))
                # remove stop words
                if stop_words:
                    filt_selected = list()
                    for sen in selected:
                        llll = []
                        for mmmm in sen:
                            if mmmm not in stop_words_cache.keys():
                                llll.append(mmmm)
                        filt_selected.append(llll)
                    selected = filt_selected
                # calculate similarity with each bucket
                if all(jaccard(seg_list, cmp_list) > args.threshold for cmp_list in selected):
                    is_match = True
                    all_bucked[bucket].append(line)
                    for w in seg_list:
                        if bucket not in p_bucket[w]:
                            p_bucket[w].append(bucket)
                    break
                # print("{} jaccard耗时 {}".format( inline, endtime - starttime))
        if not is_match:
            bucket_name = ('tmp' + id_name).format(save_idx)
            bucket_array = [line]
            all_bucked[bucket_name] = bucket_array
            for w in seg_list:
                p_bucket[w].append(bucket_name)
            save_idx += 1

    infile.close()

    batch_size = 0
    for zzzz in all_bucked:
        batch_size = batch_size + 1
        connection = pymysql.connect(host='47.99.87.74',
                                     user='august',
                                     password='august',
                                     db='august',
                                     port=33306
                                     )
        cursor = connection.cursor()

        all_bucked_data = []
        for zx in all_bucked[zzzz]:
            all_bucked_data.append([all_bucked[zzzz][0], zx, today])
        print("当前批次  {} 共 {}".format(batch_size,len(all_bucked)))
        cursor.executemany("insert into 凤巢长尾词分组(group_id,keyword,created_date) values(%s,%s,%s)",
                           (all_bucked_data))
        connection.commit()
        cursor.close()
        connection.close()


    print('All is well')


if __name__ == '__main__':
    main()
