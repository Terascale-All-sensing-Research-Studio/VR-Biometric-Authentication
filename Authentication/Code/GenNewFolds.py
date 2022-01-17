import argparse
import random
import os

parser = argparse.ArgumentParser(description='Generate New Training/Testing User Splits')
parser.add_argument('IdealUsersPerFold', type=int,
                    help='Preferred number of users per testing fold')
parser.add_argument('--ForceSameAmount', dest='ForceSame',
                    action='store_const', const=True, default=False,
                    help='Force test splits to have the same number of users (extras dropped)')
parser.add_argument('--FileToReadUsers', dest='ReadUserFile', type=str, default='<>',
                    help='Fold file to to grab users from. Allows for creation of folds with a subset of total users')
args = parser.parse_args()

if args.ReadUserFile == '<>':
    total_num_users = 41
    user_ids = [x for x in range(41)]
else:
    with open(args.ReadUserFile, 'r') as f:
        fold_file_text = f.readlines()
        f.close()
    user_ids = [int(u) for f in fold_file_text for u in f.split(',') if '\n' not in u]
    total_num_users = len(user_ids)

print(user_ids)
print(total_num_users)

assert(args.IdealUsersPerFold < total_num_users), "Can't have more users in a fold than total"
random.shuffle(user_ids)

num_folds = total_num_users//args.IdealUsersPerFold
print('Num Folds: {}'.format(num_folds))
avg_users_per_fold = total_num_users/num_folds

## We are going to wiggle the amount of folds so we best match IdealUsersPerFold
if not args.ForceSame:
    print('Avg Users per fold: {}'.format(avg_users_per_fold))
    IdealUsers = args.IdealUsersPerFold
    while abs(IdealUsers-avg_users_per_fold) > abs(IdealUsers-(total_num_users/(num_folds+1))):
        num_folds += 1
        avg_users_per_fold = total_num_users/num_folds

    print('Adjusted Num Folds: {}'.format(num_folds))
    print('Adjusted Avg Users: {}'.format(avg_users_per_fold))

cur_fold = 0
all_folds = [[] for _ in range(num_folds)]
for user_id in user_ids:
    if args.ForceSame and cur_fold == 0 and len(all_folds[0]) == args.IdealUsersPerFold:
        break
    all_folds[cur_fold] += [str(user_id)]
    cur_fold = (cur_fold + 1) % num_folds

for fold in all_folds:
    print('{}: {}'.format(len(fold), fold))

try:
    os.mkdir('NewFolds')
except:
    print('Couldn\'t make dir <NewFolds>')
    pass

with open('NewFolds/{}Fold.txt'.format(num_folds), 'w') as f:
    for fold in all_folds:
        f.write(', '.join(fold+['\n']))
    f.close()
