import os
import sys
import argparse
parser = argparse.ArgumentParser(description='Get Information to print results nicely')
parser.add_argument('Directory', type=str,
                    help='Directory with results in it')
parser.add_argument('--SysCombo', dest='SysCombo', type=str, default='Quest1Quest2',
                    help='System combination to look at results across all features')
parser.add_argument('--Fold', dest='Fold', type=str, default='',
                    help='Fold to look at')
parser.add_argument('--Test', dest='Test', type=str, default='Test0',
                    help='Test To Load. Setting this to \'All\' will print all tests.')
parser.add_argument('--Best', dest='Best', action='store_const', const=True, default=False,
                    help='Print the best test results.')

args = parser.parse_args()
assert(os.path.isdir(args.Directory)), "Provided directory path ({}) is not a directory".format(args.Directory)
all_dirs = os.listdir(args.Directory)

valid_dirs = []

FoldName = args.Fold if 'Fold' in args.Fold else args.Fold+'Fold'
SysCombo = args.SysCombo

metrics = {}
for dir_name in all_dirs:
    if FoldName in dir_name and SysCombo in dir_name:
        valid_dirs += [dir_name]    

max_key_len = 0
if args.Test == 'All':
    TestFiles = os.listdir(os.path.join(args.Directory, valid_dirs[0]))
else:
    TestFiles = [args.Test]

for valid_dir_name in valid_dirs:
    for Test in TestFiles:
        with open(os.path.join(args.Directory, valid_dir_name, Test, 'metrics.txt'), 'r') as f:
            results = f.readlines()[1]
            f.close()
        acc, eer_all, thold_all, eer_first, thold_first = (float(val.strip(' \n\t%')) for val in results.split(', '))

        if args.Best:
            system_info = valid_dir_name.rsplit('_',1)[0]
            key = (system_info, Test)
            if key in metrics:
                new_acc = max(metrics[key][0], acc)
                new_eer = min(metrics[key][1], eer_first)
                metrics[key] = (new_acc, new_eer)
            else:
                metrics[key] = (acc, eer_first)
        else:
            system_info = valid_dir_name
            metrics[(system_info, Test)] = (acc, eer_first)

        max_key_len = max(len(system_info), max_key_len)

metric_keys = list(metrics.keys())
metric_keys.sort()
for key in metric_keys:
    acc, eer = metrics[key]
    print('{1:>{0}} [{2}]: {3:6.2f}%, {4:6.2f}%'.format(max_key_len, key[0], key[1], acc, eer))
    
