from fine_row_cv_userprofile_classify import *
from metrics import metrics


METRICS_MAP = ['NDCG', "MRR", "MRR@10", "MRR@20", "MRR@50"]

featuresEncode = getFeaturesEncode()

hash_topic_id = featuresEncode["topic_id"]["hash_info"]
hash_topic_cate = featuresEncode["topic_cate"]["hash_info"]
hash_cate = featuresEncode["cate"]["hash_info"]
hash_subcate = featuresEncode["subcate"]["hash_info"]
hash_user_city = featuresEncode["user_city"]["hash_info"]
hash_gender = featuresEncode["gender"]["hash_info"]
hash_decoration_status = featuresEncode["decoration_status"]["hash_info"]
hash_wiki = featuresEncode["wiki"]["hash_info"]
hash_user_identity_type = featuresEncode["user_identity_type"]["hash_info"]
hash_obj_type = featuresEncode["obj_type"]["hash_info"]
hash_uid = featuresEncode["uid"]["hash_info"]
hash_obj_id = featuresEncode["obj_id"]["hash_info"]


def get_eval_data(train_examples, ):
    hashUserFeature, hashObjFeature = getHashData()

    hash_uid_index2id = {}
    hash_obj_index2id = {}

    for uid, index in hash_uid.items():
        hash_uid_index2id[str(int(index))] = uid

    for obj_id, index in hash_obj_id.items():
        hash_obj_index2id[str(int(index))] = obj_id

    for (ex_index, example) in enumerate(train_examples):
        docIdIndex = int(example.doc_id)
        uidIndex = int(example.uid_for_rerank)

        if str(docIdIndex) not in hash_obj_index2id or str(uidIndex) not in hash_uid_index2id:
            continue

        docId = hash_obj_index2id[str(docIdIndex)]
        uid = hash_uid_index2id[str(uidIndex)]

        hashObjFeature[docId] = {
            "doc_id" : example.doc_id,
            "favorite_num" : example.favorite_num,
            "like_num" : example.like_num,
            "comment_num" : example.comment_num,
            "score" : example.score,
            "interval_days" : example.interval_days,
            "wiki" : example.wiki,
            "obj_type" : example.obj_type,
            "position" : example.position,
            "topic_id" : example.topic_id,
            "topic_cate" : example.topic_cate,
            "cate" : example.cate,
            "subcate" : example.subcate,
            "author_uid_for_rerank" : example.author_uid_for_rerank,
            "author_type" : example.author_type,
            "author_city" : example.author_city,
            "author_gender" : example.author_gender,
            "item_like_7day" : example.item_like_7day,
            "item_comment_7day" : example.item_comment_7day,
            "item_favorite_7day" : example.item_favorite_7day,
            "item_click_7day" : example.item_click_7day,
            "item_like_30day" : example.item_like_30day,
            "item_comment_30day" : example.item_comment_30day,
            "item_favorite_30day" : example.item_favorite_30day,
            "item_click_30day" : example.item_click_30day,
            "item_exp_30day" : example.item_exp_30day,
            "author_becomment" : example.author_becomment,
            "author_befollow" : example.author_befollow,
            "author_belike" : example.author_belike,
            "author_befavorite" : example.author_befavorite,
            "item_exp_7day" : example.item_exp_7day,
        }

        isDeleteValidClickSeq = example.isDeleteValidClickSeq
        isDeleteSearchSeq = example.isDeleteSearchSeq

        valid_click_doc_id_seq = example.valid_click_doc_id_seq
        valid_click_topic_id_seq = example.valid_click_topic_id_seq
        valid_click_topic_cate_seq = example.valid_click_topic_cate_seq
        valid_click_cate_seq = example.valid_click_cate_seq
        valid_click_subcate_seq = example.valid_click_subcate_seq
        valid_click_author_city_seq = example.valid_click_author_city_seq
        valid_click_author_uid_seq = example.valid_click_author_uid_seq
        search_word_history_seq = example.search_word_history_seq

        if isDeleteValidClickSeq == 1:
            valid_click_doc_id_seq.append(example.doc_id)
            valid_click_topic_id_seq.append(example.topic_id)
            valid_click_topic_cate_seq.append(example.topic_cate)
            valid_click_cate_seq.append(example.cate)
            valid_click_subcate_seq.append(example.subcate)
            valid_click_author_city_seq.append(example.author_city)
            valid_click_author_uid_seq.append(example.author_uid_for_rerank)

        if isDeleteSearchSeq == 1:
            search_word_history_seq.append(example.queryIndex)

        hashUserFeature[uid] = {
            "uid_for_rerank" : example.uid_for_rerank,
            "user_identity_type" : example.user_identity_type,
            "gender" : example.gender,
            "age" : example.age,
            "user_city" : example.user_city,
            "decoration_status" : example.decoration_status,
            "action_count_like": example.action_count_like,
            "action_count_comment": example.action_count_comment,
            "action_count_favor": example.action_count_favor,
            "action_count_all_active": example.action_count_all_active,
            "valid_click_doc_id_seq": valid_click_doc_id_seq,
            "valid_click_topic_id_seq": valid_click_topic_id_seq,
            "valid_click_topic_cate_seq": valid_click_topic_cate_seq,
            "valid_click_cate_seq": valid_click_cate_seq,
            "valid_click_subcate_seq": valid_click_subcate_seq,
            "valid_click_author_city_seq": valid_click_author_city_seq,
            "valid_click_author_uid_seq": valid_click_author_uid_seq,
            "search_word_history_seq": search_word_history_seq,
        }

    with open(fineRowHashDataJsonFile, 'w') as f:
        fineRowHashDataJson = {
            "hash_uid_feature" : hashUserFeature,
            "hash_obj_id_feature" : hashObjFeature,
        }

        f.write(json.dumps(fineRowHashDataJson))

    print("使用的验证集 的 数据地址：" + eval_file)

    hash_eval_datas = np.load(eval_file, allow_pickle=True).item()

    eval_datas = hash_eval_datas["eval_classify_samples"]

    # 初始化
    all_metrics = np.zeros(len(METRICS_MAP))

    # 存放所有 数据
    eval_all_datas = []
    # 存放 每个query 有多少条数据
    unique_str_group_num = []
    # 计数 有多少个 query是有内容 用于最后求平均
    have_data_unique_str_num = 0

    # 合并所有数据
    for unique_str, eval_data in eval_datas.items():
        if len(eval_data) > 0:
            eval_all_datas.extend(eval_data)
            unique_str_group_num.append(len(eval_data))
            have_data_unique_str_num += 1

            # 用于控制 评价指标 使用的 搜索词数量
            if have_data_unique_str_num > FLAGS.train_pkl_examples_limit and FLAGS.train_pkl_examples_limit != -1:
              break

    # 生成模型 所需的 tf record 文件
    eval_examples = processor.get_dev_examples(eval_all_datas)

    eval_file = os.path.join(eval_data_path, "eval_classify.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, FLAGS.max_seq_length, tokenizer, eval_file, task_name)

    # 构建 tf.dataset
    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
    )

    # 预测数据
    result = estimator.predict(input_fn=eval_input_fn)

    # 记录 当前搜索词已经 拿了多少条样本 和  group_num 相同时 判断为一组数据
    have_append_data_nums = 0
    # 获取 每个搜索词 包含的  样本数
    group_num = 0

    results = []
    for index, item in enumerate(result):
        # 获取第一个搜索词的样本数
        if index == 0:
            group_num = unique_str_group_num.pop(0)

        # 保存 预测结果
        results.append((item["max_scores"], item["vals"]))

        # 当前搜索词 拿了多少条样本 + 1
        have_append_data_nums += 1

        if have_append_data_nums == group_num:
            # 同一个搜索词下 的 一组 数据计算评价指标
            max_scores, labels = zip(*results)

            # 预测 得分
            max_scores = np.array(max_scores)

            # 真实 label
            labels = np.array(labels)

            # 预测得分  从 大到小 排序 的 索引
            pred_docs_sorted_index = max_scores.argsort()[::-1]

            # 预测得分  从 大到小 排序
            scores_sorted = [max_scores[x] for x in pred_docs_sorted_index]

            # 按照 预测得分  从 大到小 排序 后。 获取 其 原本的真实标签
            pred_gt = [labels[x] for x in pred_docs_sorted_index]

            # 真实标签 从大到小 排序
            gt = sorted(labels, reverse=True)

            # 精排 目前 只针对前 200条内容 进行 排序
            metric = metrics.metrics(
                gt=gt[:200], pred=scores_sorted[:200], pred_gt=pred_gt[:200], metrics_map=METRICS_MAP)

            all_metrics += metric

            # 重置 数据
            have_append_data_nums = 0
            # print(results)
            results = []

            # 获取下一个 搜索词 的 样本数
            if len(unique_str_group_num) > 0:
                group_num = unique_str_group_num.pop(0)

    # 评价指标 求 平均值
    all_metrics /= have_data_unique_str_num