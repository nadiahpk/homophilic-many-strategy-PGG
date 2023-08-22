# a plot comparing the number of family partition structures versus the number of group compositions
# as group size n increases

from math import factorial
import matplotlib.pyplot as plt

# do the first 49
ns = [i for i in range(1, 50)]

# Number of partitions of 1 to 49, from https://oeis.org/A000041/list
# This is the number of family partition structures that need to have their
# probabilities specified to use our model
n_fams = [1,2,3,5,7,11,15,22,30,42,56,77,101,135,176,231, 297,385,490,627,792,1002,1255,1575,1958,2436,3010, 3718,4565,5604,6842,8349,10143,12310,14883,17977, 21637,26015,31185,37338,44583,53174,63261,75175, 89134,105558,124754,147273,173525]

# Number of possible groups (Jensen & Rigos 2018 Int J Game Theory) 
# This is the maximum number of group-composition probabilities that need
# to be specified in a matching model

# assuming number of strategies m = number of individuals in the group n
ms = ns
n_grps = [factorial(n+m-1)/(factorial(n)*factorial(m-1)) for n, m in zip(ns, ms)]

# assuming number of strategies m = 4
m4_n_grps = [factorial(n+4-1)/(factorial(n)*factorial(4-1)) for n in ns]

# assuming number of strategies m = 5
#m5_n_grps = [factorial(n+5-1)/(factorial(n)*factorial(5-1)) for n in ns]

plt.figure(figsize=(5, 3.8))
plt.plot(ns, n_fams, lw=2, color='blue', label='family partition structures')
plt.plot(ns, n_grps, lw=2, ls='dashed', color='red',  label='group strategy compositions ($m=n$)')
plt.plot(ns, m4_n_grps, lw=2, ls='solid', color='red', label='group strategy compositions ($m=4$)')
#plt.plot(ns, m5_n_grps, lw=2, ls='solid', color='red', label='group compositions ($m=5$)')
plt.yscale('log')
plt.ylim((1, 1e11)) # cuts off some of the line, but makes it easier to read

plt.xlabel('group size $n$', fontsize='x-large')
plt.ylabel('maximum number of\nprobabilities to be specified', fontsize='x-large')
plt.legend(loc='best')
plt.tight_layout()
#plt.show()
plt.savefig('../../results/compute_time/number_family_partns_groups.pdf')
plt.close()
