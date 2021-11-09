# import nltk
import string
import nltk
from nltk import WordNetLemmatizer
from stop_list import *
from query import Query

# Descirption = Read Query and parse queries
# Input  = FileName
# Output = Preprocessed queries
def parseQuery(filename):
  with open(filename,"r") as f:
    IDS, queries, WS = [], [], []
    new_queries = []
    query_docs = []
    cont = False
    string = ""

    for line in f:
      #print(line)
      if ".I" in line:
        cont = False
        if len(string) > 0:
          queries.append(string)
        string = ""
        part = line.split()
        IDS.append(part[1])

      if ".W" in line:
        cont = True

      if cont == True:
        string = string + line

    if len(string) > 0 :
      queries.append(string)

    for query in queries:
      query = query[2:]
      new_queries.append(query)
    length = len(IDS)

    for count in range(length):
      I = IDS[count]
      qu = new_queries[count]
      query_docs.append(Query(I,qu))
    f.close()
    return query_docs

# Description = Read Documents and parse documents
# Input  = FileName
# Output = Preprocessed documents
def parseAbsDocs(filename):
  with open(filename,"r") as f:
    abstracts = []
    string = ""
    cont = False
    for line in f:
      if ".I" in line:
        cont = False
        if len(string)>0:
          abstracts.append(string)
          string = ""
      if ".W" in line:
        cont = True
      if cont == True:
        string = string + line
    if len(string)>0:
      abstracts.append(string)
    new_abstracts = []
    for abst in abstracts:
      abst = abst[2:]
      new_abstracts.append(abst)
    f.close()
    return new_abstracts

# Description = Tokenize queries
# Input  = Queries
# Output = tokenized queries
def tokenize(docs, docType="query"):
  if docType == "query":
    qToks = []
    for query in docs:
      qToks.append(query.tokenize())
    return qToks
  else:
    absToks = []
    for abstract in docs:
      sentences = nltk.sent_tokenize(abstract)
      toks = []
      for sentence in sentences:
        stopset = [word for word in closed_class_stop_words]
        stop_punc = list(string.punctuation)
        stops = stopset+stop_punc
        tokens = nltk.wordpunct_tokenize(sentence)
        lemmatizer = WordNetLemmatizer()
        tokens = [w for w in tokens if w.lower() not in stops ]
        tokens = [w.lower() for w in tokens]
        [lemmatizer.lemmatize(w) for w in tokens]
        filtered_tokens = [x for x in tokens if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
        toks.append(filtered_tokens)
      absToks.append(toks)
    return absToks

# Description
# Input  = Tokenized documents
# Output = Inverted index
def organize(tokens, docType):
  numDocs = len(tokens)
  dic = {}
  count = 0
  for doc in tokens:
    if docType == "query":
      for tok in doc:
        if tok not in dic:
          dic[tok] = [0]*numDocs
          dic[tok][count] = 1
        else:
          dic[tok][count] = dic[tok][count] +1

    if docType == "abstract":
      for sentence in doc:
        for tok in sentence:
          if tok not in dic:
            dic[tok] = [0]* numDocs
            dic[tok][count] = 1
          else:
            dic[tok][count] = dic[tok][count] + 1
    count = count + 1
  return dic