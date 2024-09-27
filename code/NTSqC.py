# run Quality Check against new sub data

import os
import sys
import pandas as pd
import sys

def parse_cmd_args():
    import argparse
    parser = argparse.ArgumentParser(description='QC for ATS')
    parser.add_argument('-s', type=str, help='Path to submission')
    parser.add_argument('-o', type=str, help='Path to output for QC plots and Logs')
    parser.add_argument('-sub', type=str, help='Subject ID')

    return parser.parse_args()

def df(submission):
    submission = pd.read_csv(submission)
    return submission

def qc(submission):
    errors = []
    submission = df(submission)
    # Check if submission is a DataFrame
    if not isinstance(submission, pd.DataFrame):
        errors.append('Submission is not a DataFrame. Could not run QC')
    
    # Check if submission is empty
    if len(submission) == 0:
        errors.append('Submission is empty')
    
    #Check if submission has correct number of rows (within 5% of expected = 179)
    if len(submission) < 170 or len(submission) > 188:
        print(f'WARNING: Submission has incorrect number of rows. Expected 179 - found {len(submission)}')
    # If there are any errors, print them and exit
    if errors:
        for error in errors:
            print(error)
        sys.exit(1)

    print("All QC checks passed.")
        
    
def plots(submission, output, sub):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from math import pi

    df = pd.read_csv(submission)
    test = df[df['block'] == 'test']

    percentages = test.groupby('block_cond')['response'].apply(lambda x: (x != 'None').mean())

    # Plotting the circular bar graph
    def plot_circular_bar_graph(percentages, name, output_name):
        startangle = 90
        colors = ['#4393E5', '#43BAE5', '#7AE6EA', '#E5A443']
        
        # Convert data to fit the polar axis
        ys = [i *1.1 for i in range(len(percentages))]   # One bar for each block
        left = (startangle * pi * 2) / 360  # This is to control where the bar starts

        # Figure and polar axis
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plt.subplot(projection='polar')

        # Plot bars and points at the end to make them round
        for i, (block, percentage) in enumerate(percentages.items()):
            ax.barh(ys[i], percentage * 2 * pi, left=left, height=0.5, color=colors[i % len(colors)], label=block)
            ax.text(percentage + left + 0.02, ys[i], f'{percentage:.0%}', va='center', ha='left', color='black', fontsize=12)

        plt.ylim(-1, len(percentages))

        # Custom legend
        ax.legend(loc='center', bbox_to_anchor=(0.5, -0.1), frameon=True) 

        # Clear ticks and spines
        plt.xticks([])
        plt.yticks([])
        ax.spines.clear()
        plt.title(name, fontsize=15, pad=20, color="black")

        plt.savefig(os.path.join(output, f'{sub}_'+output_name+'.png'))
        plt.close()

    plot_circular_bar_graph(percentages, 'Response Rate by Conditon', 'response_rate')
    acc_perc = test.groupby('block_cond')['correct'].mean()
    plot_circular_bar_graph(acc_perc, 'Accuracy Rate by Condition', 'accuracy_rate') 

    sns.boxplot(x='block_cond', y='response_time', hue='correct', data=test, showfliers=False)
    sns.stripplot(x='block_cond', y='response_time', hue='correct', data=test, color='black', alpha=0.5)
    plt.title('Response time per block')
    plt.xlabel('Block')
    plt.ylabel('Response time')
    plt.savefig(os.path.join(output, f'{sub}_rt.png'), bbox_inches='tight')


def main():

    #parse command line arguments
    args = parse_cmd_args()
    submission = args.s
    output = args.o
    sub = args.sub

    # check if submission is a csv
    if not submission.endswith('.csv'):
        raise ValueError('Submission is not a csv')
    # check if submission exists
    if not os.path.exists(submission):
        raise ValueError('Submission does not exist')
    # run QC
    qc(submission)
    
    print(f'QC passed for {submission}, generating plots...')
    # generate plots
    plots(submission, output, sub)
    return submission
    
    
if __name__ == '__main__':
    main()



    
    


