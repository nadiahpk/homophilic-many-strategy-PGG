# make latex tables out of the csv files with the coefficients

import os
import pandas as pd


# parameters
# ---

# where the cvs files with the coefficients are stored
results_dir = '../../results/related2partprob/'

# list of group sizes we wish to make tables for
nV = list(range(3, 8))


# make a latex table for each result
# ---

# open the latex file we'll write the tables to
f = open(results_dir + 'coefficients_tables.tex', 'w')


for n in nV:

    # psi -> rho coefficients
    # ---

    fname = results_dir + 'psi2rho_numerators_n' + str(n) + '.csv'
    df = pd.read_csv(fname)

    # write the rhos as subscripts of r
    df['rho'] = df['rho'].apply(lambda x: '$r_{\{' + ','.join(sorted(str(x).split('|'), reverse=True)) + '\}}$')

    # write the common denominators as denominators
    df['common_denom'] = df['common_denom'].apply(lambda x: '$1/' + str(x) + '$')

    # get the list of psis
    psisV = list(df.columns[2:])

    # write them as subscripts of F
    FV = ['$F_{\{' + ','.join(sorted(x.split('|'), reverse=True)) + '\}}$' for x in psisV]

    # put it together with first two columns as nice things
    header = [' ', 'factor'] + FV
    df.columns = header

    caption = 'Coefficients to convert from family partition probabilities to relatedness coefficients for $n=' + str(n) + '$.'
    label = 'psi2rho' + str(n)
    ltx = df.to_latex(index=False, escape=False, column_format='l' + 'c'*(len(FV)+1), caption=caption, label=label)

    # tidy up latex directly
    ltx = ltx.replace("\\begin{table}", "\\begin{table}[h]")

    f.write(ltx)
    f.write('\n')


    # rho -> psi coefficients
    # ---

    fname = results_dir + 'rho2psi_coeffs_n' + str(n) + '.csv'
    df = pd.read_csv(fname)

    # write the psis as subscripts of F
    df['psi'] = df['psi'].apply(lambda x: '$F_{\{' + ','.join(sorted(str(x).split('|'), reverse=True)) + '\}}$')

    # get the list of rhos
    rhosV = list(df.columns[2:])

    # write them as subscripts of r
    rV = ['$r_{\{' + ','.join(sorted(x.split('|'), reverse=True)) + '\}}$' for x in rhosV]

    # put it together with first two columns as nice things
    header = ['', 'factor'] + rV
    df.columns = header

    caption = 'Coefficients to convert from relatedness coefficients to family partition probabilities for $n=' + str(n) + '$.'
    label = 'rho2psi' + str(n)
    ltx = df.to_latex(index=False, escape=False, column_format='l' + 'c'*(len(rV)+1), caption=caption, label=label)

    # tidy up latex directly
    ltx = ltx.replace("\\begin{table}", "\\begin{table}[h]")

    f.write(ltx)
    f.write('\n')

f.close()
