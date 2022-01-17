import numpy as np
from sklearn.metrics import roc_curve

import os
import sys

import math
import itertools

def eer_rate(y_pred, y_true):
    fpr, tpr, tHolds = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    EER = (fpr[idx]+fnr[idx])/2
    tHold = tHolds[idx]
    pos = np.count_nonzero(y_true)
    neg = len(y_true) - pos
    tp = int(tpr[idx] * pos)
    fn = int(fnr[idx] * pos)
    fp = int(fpr[idx] * neg)
    tn = int(tnr[idx] * neg)
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denom < 0.000001:
        MCC = 0
    else:
        MCC = ((tp*tn)-(fp*fn))/denom
    return EER, tHold, MCC

def load_results(filename, sys_pred, sys_true, u1_throws, u2_throws, user_group):
    with open(filename, 'r') as f:
        full_txt = f.readlines()
        f.close()
    throw_id = 0
    record_id = 0
    users_in_fold = []
    min_dist = 1000000000.0
    cur_best = 0
    correct = 0
    total = 0
    missed = 0
    kept = 0
    skip_acc = True 
    for line in full_txt:
        data = line.split(',')
        if data[0] == '':
            if not skip_acc:
                if cur_best == prior_user:
                    correct += 1
                total += 1
            min_dist = 1000000000.0
            skip_acc = True
            continue

        # We only want users in user_group
        # Useful for parsing out only test-data pairs or all-data pairs
        if int(data[0]) not in user_group \
           or int(data[2]) not in user_group:
            missed += 1
            continue
        if int(data[1]) not in u1_throws \
           or int(data[3]) not in u2_throws:
            continue
        kept += 1 

        network_value = float(data[4])
        skip_acc = False
        prior_user = int(data[0])
        if min_dist > network_value:
            cur_best = int(data[2])
            min_dist = network_value

        user_index = user_group.index(int(data[0]))
        sys_pred[user_index, record_id, throw_id] = 1.0 - network_value #flip from min to max
        sys_true[user_index, record_id, throw_id] = int(int(data[0]) == int(data[2]))

        if user_index not in users_in_fold:
            users_in_fold += [user_index]

        throw_id += 1
        if throw_id % sys_pred.shape[-1] == 0:
            record_id += 1
            throw_id = 0
        if record_id % sys_pred.shape[-2] == 0:
            record_id = 0

    if total == 0:
        total = 1
    #print('\t\t{} Skipped | {} Looked at'.format(missed, kept))
    return users_in_fold, correct/total*100.0
   
def write_metrics(out_filepath, sys_pred, sys_true, user_ids, acc):
    u_pred = np.reshape(sys_pred[user_ids], (-1))
    #print(u_pred.shape)
    u_true = np.reshape(sys_true[user_ids], (-1))
    EER_all, tHold_all, MCC_all = eer_rate(u_pred, u_true)

    u_pred = np.reshape(np.max(sys_pred[user_ids], axis=-1), (-1))
    u_true = np.reshape(sys_true[user_ids, :, 0], (-1))
    EER_first, tHold_first, MCC_first = eer_rate(u_pred, u_true)

    with open(out_filepath, 'w') as f:
        f.write('Accuracy, EER (All), THold (All), EER (Closest), THold (Closest)\n')
        f.write(
            '{:6.2f}, {:6.2%}, {:6.2f}, {:6.2%}, {:6.2f}\n'.format(
             acc, EER_all, tHold_all, EER_first, tHold_first))
        f.close()


def getMetrics(in_dir, out_dir, foldPath, epoch,
               u1_throws=[i for i in range(10)], 
               u2_throws=[i for i in range(10)],
               num_users=41, user_group=[],
               model_id=1, intrafold_only=False):
        
    if user_group == []:
        user_group = [x for x in range(num_users)]
    else:
        num_users = len(user_group)


    sys_pred = np.zeros((num_users, len(u1_throws)*num_users, len(u2_throws)))
    sys_true = np.zeros((num_users, len(u1_throws)*num_users, len(u2_throws)))


    accuracy = 0

    ## Folds results are stored in directories under <in_dir>
    ## These are intergers 0->NumFolds-1
    ## We want them in order since it makes things easier
    folds = [directory for directory in os.listdir('{}/'.format(in_dir)) 
                        if os.path.isdir(os.path.join(in_dir, directory))]
    folds.sort(key=lambda x: int(x))

    with open(foldPath, 'r') as f:
        users_txt = f.readlines()
        f.close()
    fold_users = [[u for u in users.split(',') if '\n' not in u] for users in users_txt]


    intrafold_start = 0
    total_computed_users = 0
    intrafold_pred = []
    intrafold_true = []
    for fold_id, _ in enumerate(fold_users):
        print('\t\t{}'.format(fold_id))
        fold_name = folds[fold_id]
        
        if intrafold_only: ## We only want to look at test user pairs
            user_group = list(range(intrafold_start, intrafold_start + len(fold_users[fold_id])))
            intrafold_start += len(fold_users[fold_id])
            sys_pred = np.zeros((len(user_group), len(u1_throws)*len(user_group), len(u2_throws)))
            sys_true = np.zeros((len(user_group), len(u1_throws)*len(user_group), len(u2_throws)))

        fold_in_dir = os.path.join(in_dir, fold_name)
        fold_out_dir = os.path.join(out_dir, str(fold_id))

        try:
            os.mkdir('{}'.format(fold_out_dir))
        except:
            pass

        good_users = 0
        
        users_in_fold, acc = load_results(os.path.join(fold_in_dir, 'FoldAcc{}.csv'.format(epoch)),
                                            sys_pred, sys_true, u1_throws, u2_throws,
                                            user_group)
        accuracy += (acc*len(users_in_fold))

        if len(users_in_fold) != 0:
                for user_id in users_in_fold:
                    write_file = os.path.join(fold_out_dir, 'user{}_metrics.txt'.format(user_group[user_id]))
                    write_metrics(write_file, sys_pred, sys_true, user_id, acc)
                write_metrics(os.path.join(fold_out_dir, 'fold_metrics.txt'), sys_pred, sys_true, users_in_fold, acc)
                good_users = len(users_in_fold)

        if intrafold_only:
            intrafold_pred += [sys_pred]
            intrafold_true += [sys_true]

        total_computed_users += good_users

    accuracy = accuracy / float(num_users)
    if intrafold_only:
        for data_idx in range(len(intrafold_pred)):
            intrafold_pred[data_idx] = intrafold_pred[data_idx].reshape(-1, len(u2_throws))
            intrafold_true[data_idx] = intrafold_true[data_idx].reshape(-1, len(u2_throws))
        sys_pred = np.concatenate(intrafold_pred, axis=0)
        sys_true = np.concatenate(intrafold_true, axis=0)
    else:
        sys_pred = np.reshape(sys_pred, (-1, len(u2_throws)))
        sys_true = np.reshape(sys_true, (-1, len(u2_throws)))

    print('\t\tMy Computed: {} | Correct: {}'.format(total_computed_users, num_users))

    u_pred = np.reshape(sys_pred, (-1))
    u_true = np.reshape(sys_true, (-1))
    EER_all, tHold_all, MCC_all = eer_rate(u_pred, u_true)
    
    u_pred = np.reshape(np.max(sys_pred, axis=1), (-1))
    u_true = np.reshape(sys_true[:, 0], (-1))
    EER_first, tHold_first, MCC_first = eer_rate(u_pred, u_true)
    
    with open(os.path.join(out_dir, 'metrics.txt'), 'a') as f:
        f.write('Accuracy, EER (All), THold (All), EER (First), THold (First)\n')
        f.write(
            '{:#7.2f}%, {:#9.2%}, {:#11.2f}, {:#11.2%}, {:#13.2f}\n'.format(
             accuracy, EER_all, tHold_all, EER_first, tHold_first))
    
        f.close()



#if __name__=='__main__':
#    in_dir=sys.argv[1]
#    out_dir=sys.argv[2]
#    foldPath = sys.argv[3]
#    epoch=int(sys.argv[4])
#
#    getMetrics(in_dir, out_dir, foldPath, epoch,
#               u1_throws=[i for i in range(10)], 
#               u2_throws=[i for i in range(10)],
#               num_users=41, user_group=[],
#               model_id=1, intrafold_only=False):



