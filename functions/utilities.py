import os
import pandas as pd

def write_fixedpts_latex(results_dir, file_prefix):
    '''
    A utility function that takes information about the steady states stored in csv files
    and writes the information to latex tables. 

    Inputs:
    ---

    results_dir, str

        A directory containing fixed-point files and also the directory where the latex tables 
        will be saved, e.g., '../../results/sigmoid_UDCL'

    file_prefix, str

        A prefix shared by all csv files to be included
        e.g., 'fixedpts_stability_sigmoidUDCL_v1_leader_driven_ngrid_9'

    Outputs:
    ---

    ltx_fname, str

        The filename to which the latex tables were written: [file_prefix].tex
    '''

    # open a latex file to write the tables to
    # ---

    ltx_fname = os.path.join(results_dir, file_prefix + '.tex')
    f = open(ltx_fname, 'w')


    # get a list of all files with the relevant prefix
    # ---

    len_prefix = len(file_prefix)
    all_files = os.listdir(results_dir)
    fp_files = [fname for fname in all_files if fname[:len_prefix] == file_prefix and fname[-4:] == '.csv']


    # turn each file's content into a latex table
    # ---

    for fname in fp_files:

        # read in the steady states information from this file
        fname_full = os.path.join(results_dir, fname)
        df = pd.read_csv(fname_full)

        # get the strategy names out of the header
        strat_names = [v[-1:] for v in df.columns if v[:7] == 'inv_fit']

        # use the strategy names to create headers for the latex table

        # will split table into multicolumn with: steady state, stability, invasion fitness
        stdy_columns = ['p_' + strat_name + '*' for strat_name in strat_names]
        ifit_columns = ['inv_fit_' + strat_name for strat_name in strat_names]

        # create nice headers for each
        stdy_headers = ['$p_' + strat_name + '^*$' for strat_name in strat_names]
        ifit_headers = [strat_name for strat_name in strat_names]

        # caption will be the file name
        caption = 'Steady states from ' + r'{\tt ' + fname + '}'
        caption = caption.replace('_', '\_') # escape underscores in file names

        # reshape the dataframe so we can easily make a table with multicolumn headings
        # Helpful example here: 
        # https://tex.stackexchange.com/questions/579468/generating-a-latex-table-with-multiple-column-levels-using-python-pandas

        # split into its multicolumns
        df_stdy = df[stdy_columns]
        df_stab = df['stability']
        df_ifit = df[ifit_columns]

        # name each's headers nicely
        df_stdy.columns = stdy_headers
        df_ifit.columns = ifit_headers

        # bring them together
        df = pd.concat({'steady state': df_stdy, "steady state's": df_stab, 'invasion fitness of': df_ifit}, axis=1)

        # remove the index column
        n_rows, n_cols = df.shape
        df = df.set_index([[""] * n_rows])

        # turn the dataframe into a latex table
        ltx = df.to_latex(index=True, escape=False, column_format='c'*n_cols, multicolumn_format='c', caption=caption)

        # remove the empty column at the start (the index column)
        ltx = ltx.replace("{} &", "")

        # remove the NaNs and nas
        ltx = ltx.replace("NaN", "   ")
        ltx = ltx.replace("na", "  ")

        # replace the 1s and 0s with simply "1" and "0"
        ltx = ltx.replace('1.000000 ', '1        ')
        ltx = ltx.replace('0.000000 ', '0        ')
        ltx = ltx.replace('1.00000 ', '1       ')
        ltx = ltx.replace('0.00000 ', '0       ')
        ltx = ltx.replace('1.0000 ', '1      ')
        ltx = ltx.replace('0.0000 ', '0      ')
        ltx = ltx.replace('1.000 ', '1     ')
        ltx = ltx.replace('0.000 ', '0     ')
        ltx = ltx.replace('1.00 ', '1    ')
        ltx = ltx.replace('0.00 ', '0    ')
        ltx = ltx.replace('1.0 ', '1   ')
        ltx = ltx.replace('0.0 ', '0   ')

        # use the "h" option so the table prints "here" in the latex document
        ltx = ltx.replace("\\begin{table}", "\\begin{table}[h]")

        # write this table to the file
        f.write(ltx)
        f.write('\n')


    # close the latex file
    f.close()

    return ltx_fname


def partitionInteger(n):
    '''
    Find all integer partitions of n

    copied verbatim from: http://jeromekelleher.net/generating-integer-partitions.html
    '''

    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
