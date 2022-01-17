import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Get Information to print results nicely')
parser.add_argument('InDir', type=str,
                    help='Directory with network outputs (final output <test directory>)')
parser.add_argument('Filename', type=str,
                    help='Filename to write to')

args = parser.parse_args()
assert(os.path.isdir(args.InDir))

foldDirs = [x for x in os.listdir(args.InDir) if os.path.isdir(os.path.join(args.InDir, x))]

individual_user_accs = []
individual_user_eers = []

for foldDir in foldDirs:
    foldPath = os.path.join(args.InDir, foldDir)
    user_metrics = [x for x in os.listdir(foldPath) if 'user' in x]
    for user_metric in user_metrics:
        user_result_file = os.path.join(foldPath, user_metric)
        with open(user_result_file, 'r') as f:
            metrics = f.readlines()[-1]
        acc, _, _, eer, _ = (float(x.strip('%')) for x in metrics.split(', '))
        individual_user_accs += [acc]
        individual_user_eers += [eer]

print('Retrieved {} user\'s accuracy'.format(len(individual_user_accs)))
print('Retrieved {} user\'s EER'.format(len(individual_user_eers)))
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_ylabel('Accuracy', color=color)
ax1.tick_params(axis='y', colors=color)
ax1.boxplot(individual_user_accs, positions=[1],
            boxprops=dict(color=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color))

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('EER', color=color)
ax2.tick_params(axis='y', colors=color)
ax2.boxplot(individual_user_eers, positions=[2], 
            boxprops=dict(color=color), 
            whiskerprops=dict(color=color),
            capprops=dict(color=color))

fig.tight_layout()
plt.savefig(args.Filename)
