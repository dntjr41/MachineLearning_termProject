# Import Class Libraries
import pandas as pd
import numpy as np


###################################################################
# Read File

# Cran.all.1400.txt:
# Number of documents – 1,400 documents
# Fields - (i) I: document id
#         (ii) T: short description of documents
#        (iii) A: author
#         (iv) B: book
#          (v) W: content of documents

# Cran.qry.txt:
# Number of queries – 225 queries
# Field - (i) I: query id
#        (ii) W: content of query

# Cranqrel.txt:
# relevant documents for each query (target file). It contains series of query order,
# relevant document id and relevance between query and document.
# Field – (i) Query order: order of appearance of the query in cran.qry
#        (ii) Relevant document id
#       (iii) Relevance between query and document range 1 to 5
#             (As the number goes from 1 to 5, it means that the relevance decrease)

document = 'cran.all.1400.txt'
query = 'cran.qry.txt'
qrel = 'cranqrel.txt'

