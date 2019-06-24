import numpy as np
import itertools
import pandas as pd
from collections import Counter

def findRowCols(data, rowselectionsize=16, colselectionsize=10, searchingsize=18):

  data_sorted = np.sort(data, axis=0)
  index_sorted = np.argsort(data, axis=0)

  delta = np.empty((1, data_sorted.shape[1]))
  for i in np.arange(data_sorted.shape[0] - rowselectionsize + 1):
      delta = np.vstack((delta, data_sorted[i + rowselectionsize - 1, :] - data_sorted[i, :]))
  delta = np.delete(delta, 0, axis=0)

  res = np.empty((rowselectionsize, data.shape[1]))  # res is row combinations of rows index
  for col, start in zip(range(data.shape[1]), np.argmin(delta, axis=0)):
      res[:, col] = index_sorted[start:start + rowselectionsize, col]

  row_cnt = Counter(res.reshape(-1).astype(int))
  top_searchingsize_rows = [a for (a, _) in row_cnt.most_common(data.shape[1])[:searchingsize]]

  maxdelta = {}
  cols = {}
  for rows in itertools.combinations(top_searchingsize_rows, rowselectionsize):
      top_16_data = data[np.array(rows)]
      col_delta = pd.Series(np.ptp(top_16_data, axis=0))
      x = col_delta.nsmallest(colselectionsize)
      cols[rows] = list(x.index.values)
      maxdelta[rows] = x.iloc[-1]

  Maxdelta = pd.Series(maxdelta).nsmallest(1).values[0]
  Rows = list(list(pd.Series(maxdelta).nsmallest(1).keys())[0])
  Columns = cols[tuple(Rows)]
  
  return Maxdelta, np.sort(Rows), np.sort(Columns)


def make_selection_mask(data, rowselectionsize = 16, colselectionsize = 10, searchingsize = 21):

    d, R, C = findRowCols(data, rowselectionsize = rowselectionsize, colselectionsize = colselectionsize, searchingsize = searchingsize)
    
    selection = np.zeros((32,32))
    selection[R[:,np.newaxis],C] = 1
    
    np.savetxt("./selection_mask.csv", selection,  delimiter=",")
    
    return selection


if __name__ == "__main__":
    # experimental data will be provided upon request
    data = np.genfromtxt("../data/initial_data.csv", delimiter=",")
    # to partition a subarray of size 16 rows and 10 columns
    selection_mask = make_selection_mask(data, rowselectionsize = 16, colselectionsize = 10, searchingsize = 21)

    
 
