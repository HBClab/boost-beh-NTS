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
    from matplotlib import patches as mpatches
    import seaborn as sb
    from math import pi

    df = pd.read_csv(submission)
    test = df[df['block'] == 'test']

    percentages = test.groupby('block_cond')['response'].apply(lambda x: (x != 'None').mean())

    # Map the 'correct' column to more descriptive labels
    test['correct_label'] = test['correct'].map({0: 'Incorrect', 1: 'Correct'})

    plt.figure(figsize=(10, 6))

    # Plot the scatterplot with hue to differentiate correct and incorrect trials
    sb.stripplot(
        x='block_cond',
        y='response_time',
        data=test,
        hue='correct_label',
        alpha=0.5,
        dodge=True,
        palette={'Correct': 'green', 'Incorrect': 'red'}
    )

    # Overlay the boxplot without the hue
    sb.boxplot(
        x='block_cond',
        y='response_time',
        data=test,
        whis=np.inf,
        linewidth=0.5,
        color='gray'
    )

    # Calculate the mean response time for each block_c
    means = test.groupby('block_cond')['response_time'].mean()

    # Calculate the mean response time for blocks 'A' and 'B'
    mean_A_B = means[['A', 'B']].mean()

    # Calculate the Mixing Cost for block 'C'
    mixing_cost = means['C'] - mean_A_B

    # Create labels for the legend with the mean values and Mixing Cost
    labels = [f'block_c {cond}: Mean = {mean:.2f}' for cond, mean in means.items()]
    labels.append(f'Mixing Cost = {mixing_cost:.2f}')

    # Create dummy handles for the legend entries
    handles = [mpatches.Patch(color='white') for _ in labels]

    # Add the legend outside the plot area
    plt.legend(handles=handles, labels=labels, title='Means and Mixing Cost by block_c',
            bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    plt.title('Response Time by block_c')

        # Adjust the layout to accommodate the legend
    plt.tight_layout()
    plt.savefig(os.path.join(output, f'{sub}_response_time.png'))
    plt.close()

    # Plot accuracy as bar chart by condition, counts of correct and incorrect trials
    plt.figure()
    sb.barplot(x='block_cond', y='correct', data=test)
    plt.title('Accuracy by block_cond')
    plt.savefig(os.path.join(output, f'{sub}_acc.png'))
    plt.close()



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



    
    


