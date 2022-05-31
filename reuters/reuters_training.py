#evaluation
#1
class_hvs = []
D = 5000
for i in range(22):
    class_hvs.append(list(([0.] * D, [0.] * D)))

for i in range(len(encoded_train[0])):
    for j in list(np.unique(y_test)):
        if y_train[i]==j:
            class_hvs[j][0] = [m+n for m,n in zip(class_hvs[j][0], encoded_train[i])]

for j in list(np.unique(y_test)):
    current = j
    for i in list(np.unique(y_test)):
        if i!=current:
            class_hvs[current][1] = [m+n for m,n in zip(class_hvs[current][1], class_hvs[i][0])]

import copy
from copy import deepcopy
class_norms = [np.linalg.norm(hv) for hv in class_hvs]
class_hvs_best = deepcopy(class_hvs)
class_norms_best = deepcopy(class_norms)


labels_train = []
for i in range(len(encoded_train)):
    if y_train == lbl:
        labels_train.append(1)
    else:
        labels_train.append(0)

# for i in range(len(class_hvs)):
#     for j in range(len(class_hvs[i])):
#         class_hvs[i][0] = class_hvs[i][0]/class_norms[i]
#         class_hvs[i][1] = class_hvs[i][1]/class_norms[i]

#3

# for i in range(len(class_hvs)):
#     for j in range(len(class_hvs[i])):
#         class_hvs[i][0] = class_hvs[i][0]/class_norms[i]
#         max_ = max(class_hvs[i][0])
#         class_hvs[i][1] = [max_-np.abs(class_hvs[i][0][j]) for j in range(len(class_hvs[i][0]))]
#0.04 all 1
# for i in range(len(class_hvs)):
#     for j in range(len(class_hvs[i])):
#         class_hvs[i][0] = class_hvs[i][0]/class_norms[i]
#         avg_ = np.mean(class_hvs[i][0])
#         class_hvs[i][1] = [avg_-np.abs(class_hvs[i][0][j]) for j in range(len(class_hvs[i][0]))]
#0.95 all 0

# for i in range(len(class_hvs)):
#     for j in range(len(class_hvs[i])):
#         class_hvs[i][0] = class_hvs[i][0]/class_norms[i]
#         avg_ = np.mean(class_hvs[i][0])
#         class_hvs[i][1] = [avg_-np.abs(class_hvs[i][0][j]) for j in range(len(class_hvs[i][0]))]
#0.95 all 0

# sign
for i in range(len(class_hvs)):
    for j in range(len(class_hvs[i])):
        class_hvs[i][0] = class_hvs[i][0]/class_norms[i]
        class_hvs[i][1] = class_hvs[i][1]/class_norms[i]
        for k in range(len(class_hvs[i][0])):
            if class_hvs[i][0][k] >0:
                class_hvs[i][0][k] = 1
            else:
                class_hvs[i][0][k] = 0

        for k in range(len(class_hvs[i][1])):
            if class_hvs[i][0][k] > 0:
                class_hvs[i][1][k] = 0
            else:
                class_hvs[i][1][k] = 1

#predict = max_match(class_hvs[0], lbl, encoded_test[0])
def max_match(class_hvs, lbl, x, class_norms):
    normalized = []
    max_score = -np.inf
    max_index = -1
    score_pos = np.dot(class_hvs[lbl][0], x) / class_norms[lbl]
    score_neg = np.dot(class_hvs[lbl][1], x) / class_norms[lbl]
    #print(score_pos, score_neg)
    if score_pos > score_neg:
        #print(score_neg)
        return 0
        #print(score_neg)
    return 1

def cal(lbl):
    labels = []
    for i in range(len(y_test)):
        if y_test[i] == lbl:
            labels.append(1)
        else:
            labels.append(0)
    #labels.count(1)
    epoch = 100
    if epoch > 0:
        class_norms = [np.linalg.norm(hv) for hv in class_hvs]
        acc_max = -np.inf
        #if log: print('\n\n' + str(epoch) + ' retraining epochs')
        for i in range(epoch):
            # if log:
            #     sys.stdout.write('epoch ' + str(i) + ': ')
            #     sys.stdout.flush()
            for j in range(len(encoded_train)):
                predict = max_match(class_hvs, lbl, encoded_train[j], class_norms)
                #predict = max_match(class_hvs, train_enc_hvs[j], class_norms)
                if predict != labels_train[j]:
                    class_hvs[predict][0] -= np.multiply(alpha/(1. + epoch/5.), encoded_train[j])
                    class_hvs[predict][1] += np.multiply(alpha/(1. + epoch/5.), encoded_train[j])
                    class_hvs[int(labels_train[i])][0] += np.multiply(alpha/(1. + epoch/5.), encoded_train[j])
                    class_hvs[int(labels_train[i])][1] -= np.multiply(alpha/(1. + epoch/5.), encoded_train[j])
            class_norms = [np.linalg.norm(hv) for hv in class_hvs]
            correct = 0
            predicted = []
            for j in range(len(encoded_test)):
                predict = max_match(class_hvs, lbl, encoded_test[j], class_norms)
                #predict = max_match(class_hvs, validation_enc_hvs[j], class_norms)
                if predict == labels[j]:
                    correct += 1
                else:
                    class_hvs[predict][0] -= np.multiply(alpha/(1. + epoch/5.), encoded_test[j])
                    class_hvs[int(labels[i])][0] += np.multiply(alpha/(1. + epoch/5.), encoded_test[j])
                    class_hvs[predict][1] -= np.multiply(alpha/(1. + epoch/5.), encoded_test[j])
                    class_hvs[int(labels[i])][1] += np.multiply(alpha/(1. + epoch/5.), encoded_test[j])
                predicted.append(predict)
            acc = float(correct)/len(encoded_test)
            # if log:
            #     sys.stdout.write("%.4f " %acc)
            #     sys.stdout.flush()
            if acc > acc_max:
                acc_max = acc
                class_hvs_best = deepcopy(class_hvs)
                class_norms_best = deepcopy(class_norms)
    accuracy = accuracy_score(labels, predicted)
    recall = recall_score(labels, predicted, average='weighted')
    precision = precision_score(labels, predicted, average='weighted')
    auc = precision_score(labels, predicted, average='weighted')
    print(accuracy, recall, precision, auc)
    cm = confusion_matrix(labels, predicted)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    print("TP, FP, FN, TN: ", TP, FP, FN, TN)
    return class_hvs
cal(0)


orthogonality = []
for i in range(len(class_hvs)):
    orthogonality.append(np.dot(class_hvs[i][0], class_hvs[i][1]))
