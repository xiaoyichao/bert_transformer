


query_qieci_path = "./data/term_weight_query_qieci.txt"
labels_path = "./data/term_weight_labels.txt"


def read_data(query_qieci_path, labels_path):
    query_list = []
    labels_list = []
    with open(query_qieci_path, 'r') as f:
        query_qieci = f.read().split("\n")
        for qieci in query_qieci:
            qieci = qieci.split(" ")
            query_list.append(qieci)


    with open(labels_path, 'r') as f:
        labels = f.read().split("\n")
        for label in labels:
            label = list(label)
            labels_list.append(label)

    return query_list,  query_list


query_list,  query_list = read_data(query_qieci_path, labels_path)
              