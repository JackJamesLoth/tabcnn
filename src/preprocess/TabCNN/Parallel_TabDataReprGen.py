####################################################################################################################################
# NOTE: Code was taken from TabCNN GitHub repo: https://github.com/andywiggins/tab-cnn/blob/master/data/Parallel_TabDataReprGen.py #
# I did have to modify it a bit, otherwise it seems to skip some fo the data                                                       #
####################################################################################################################################
from TabDataReprGen import main
from multiprocessing import Pool
import sys

# number of files to process overall
num_filenames = 360
modes = ["c", "m", "cm", "s"]

# Python 3 requires the use of a list comprehension for multiplying a list
filename_indices = [i for i in range(num_filenames) for _ in range(4)]
mode_list = modes * 360

# Ensure main function in TabDataReprGen is prepared to handle tuple arguments
# i.e., def main(args): where args is a tuple (filename_index, mode)

if __name__ == "__main__":
    # number of processes will run simultaneously
    pool = Pool(11)

    # Python 3 zip returns an iterator, so it's directly usable by pool.map without conversion
    results = pool.map(main, zip(filename_indices, mode_list))
