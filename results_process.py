import os
import pandas as pd
import matplotlib.pyplot as plt

MAX_VAL_BATCH = 10
NAX_EXAMPLE_BATCH = 9


if __name__ == "__main__":
    df = pd.read_csv(os.path.join('results', 'wandb_batch_sweep.csv'))
    df.dropna(subset=['score'], inplace=True)
    df = df[['example_batch', 'val_batch', 'score']]

    # plot of score vs. example_batch for all val_batch
    for i in range(1, MAX_VAL_BATCH + 1):
        val_df = df.loc[df.val_batch == i]
        val_df = val_df.sort_values(['example_batch'])
        plt.plot(val_df.example_batch.values, val_df.score.values, marker='o', ms=4, label=i)
    plt.title('Score vs. Example Samples')
    plt.xlabel('Number of Example Samples')
    plt.xticks(range(1, 11))
    plt.yticks([-0.01 * y for y in range(0, 27, 2)])
    plt.ylabel('Score')
    plt.legend(title='Evaluated samples', title_fontsize=8, fontsize=7)
    plt.grid()
    plt.show()

    # plot of score vs. val_batch for all example_batch
    for i in range(0, NAX_EXAMPLE_BATCH + 1):
        val_df = df.loc[df.example_batch == i]
        val_df = val_df.sort_values(['val_batch'])
        plt.plot(val_df.val_batch.values, val_df.score.values, marker='o', ms=4, label=i)
    plt.title('Score vs. Evaluated Samples')
    plt.xlabel('Number of Evaluated Samples')
    plt.xticks(range(1, 11))
    plt.yticks([-0.01 * y for y in range(0, 27, 2)])
    plt.ylabel('Score')
    plt.legend(title='Example samples', title_fontsize=8, fontsize=7)
    plt.grid()
    plt.show()
