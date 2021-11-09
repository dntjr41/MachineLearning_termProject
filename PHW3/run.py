from sklearn.metrics.pairwise import cosine_similarity
from infoRetrieval import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
nltk.download('wordnet')
import warnings
warnings.filterwarnings('ignore')

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
#####################################################################

#####################################################################
# Small Search Engine
# 1. Read file (documents, queries, cranqrel(target))
#
# 2. Text Preprocessing
#    Tokenize document and query using nltk.word_tokenize()
#    Get stopwords
#    Stemming using WordNetLemmatizer()
#    Remove punctuation marks, stopwords, frequent tokens
#
# 3. Make Inverted Index
#    Make the data frame representing the document of words
#    that have been text preprocessed
#    The overlapping words add up the data frame.
#    -> Create the final inverted index
#    Sort dataframe(words) in order
#
# 4. Find Best parameter - K(number of documents),
#                          E(similarity e to the query),
#    Select the some queries
#    Calculate cosine similarity
#    -> Using TfdifVectorizer() and cosine_similarity
#    Compare precision, recall, and F1 score using cranqrel
#    Find the best F1 score about various combinations
#    -> return best combination

def run():
  print("Process Query")
  qDocs = parseQuery("Cranfield_file/cran.qry")
  qToks = tokenize(qDocs, "query")
  qDic = organize(qToks, "query")

  print("Process Abstract Docs")
  absDocs = parseAbsDocs("Cranfield_file/cran.all.1400")
  absToks = tokenize(absDocs, "abstract")
  absDic = organize(absToks, "abstract")

  # inverted index
  absDic = sorted(absDic.items())
  inverted_index = []

  id = 1
  for i in absDic:
    count_values=i[1]
    doc_ids = []
    for index in range(len(count_values)):
      if count_values[index]>0:
        doc_ids.append(index+1)
    row=[id,i[0],doc_ids] # i[0] is word / doc_ids is document id
    #print('row: ',row)
    inverted_index.append(row)
    id+=1

  # testing
  testing_num=1
  queries=[]
  docs=[]
  queries_toks = []
  q_d_cos_sim = []
  tfidf_vectorizer = TfidfVectorizer()

  for i in range(testing_num):
    queries.append(qDocs[i].query)
    queries_toks.append(qToks[i])
  # print('queries_toks: ',queries_toks)

  for toks_index in range(len(queries_toks)):
    for tok in queries_toks[toks_index]:
      for word_in_inverted in inverted_index:
        if tok==word_in_inverted[1]:
          for doc_id in word_in_inverted[2]:
            # print('doc_id: ',doc_id)
            # docs_toks.append([toks_index,doc_id,absToks[doc_id-1]]) # query token index, document id, document content
            docs.append([doc_id,absDocs[doc_id-1]])
            q=queries[toks_index]
            d=absDocs[doc_id-1]
            vector_matrix=tfidf_vectorizer.fit_transform([q,d])
            feature_names=tfidf_vectorizer.get_feature_names()
            dense=vector_matrix.todense()
            denselist=dense.tolist()
            df=pd.DataFrame(denselist,columns=feature_names)

            #print('---df---')
            #print(df)

            cosine_similarity_matrix=pd.DataFrame(cosine_similarity(vector_matrix))
            #print('---cosine_similarity_matrix---')
            #print(cosine_similarity_matrix)
            cos_sim=cosine_similarity_matrix[0][1]

            #print('cos_sim: ',cos_sim)
            q_d_cos_sim.append([toks_index,doc_id,cos_sim]) # query index(0,1,2,3,4) / doc_id / cosine similarity

  print('===q_d_cos_sim===')
  print(q_d_cos_sim)

  # compare with cranqrel
  # cranqrel
  cranqrel=[]
  with open("Cranfield_file/cranqrel", "r") as file:
    lines=file.readlines()
    for line in lines:
      splited=line.split()
      qo=splited[0] # query order
      document=splited[1]
      cranqrel.append([qo,document])

  number_of_documents =  [5, 10, 15, 20, 25]
  similarity_query = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
  precision_recall=[]
  q_d_cos_sim.sort(key=lambda x:x[2],reverse=True)
  # print('sorted q_d_cos_sim: ')
  # print(q_d_cos_sim)
  query_index=0
  q_d_css=[] # query index에 따른 값들 저장

  for num in range(testing_num):
    if num>=testing_num:
      break

    for k in number_of_documents:
      for e in similarity_query:
        q_d_css=[]
        answer=[]

        for q_d_cs in q_d_cos_sim:
          if q_d_cs[0]!=num:
            break
          else:
            q_d_css.append(q_d_cs)

        q_d_css.sort(key=lambda x:x[2],reverse=True)
        q_d_css = pd.DataFrame(q_d_css)
        q_d_css.drop_duplicates(inplace=True)
        predict=q_d_css.iloc[0:k,1]

        print('===predict===')
        print(predict)

        for value in cranqrel:
          if value[0]==str(num+1):
            answer.append(value)
        answer=pd.DataFrame(answer)
        answer=answer.iloc[:,1]

        print('===answer===')
        print(answer)
        query_index+=1
        tp=0

        for p in predict:
          if p in answer:
              tp+=1
        precision=round(tp/len(predict),3)

        print('precision: ',precision)
        recall=round(tp/len(answer),3)

        print('recall: ',recall)
        f1_score = (precision*recall)/(precision+recall)

        print('F1 score: ',f1_score)
        precision_recall.append([k,e,precision,recall,f1_score])

    print('===precision_recall===')
    print(precision_recall)
    precision_recall_df = pd.DataFrame(precision_recall, columns=['k','e','precision','recall','f1 score'])

    print('===Best Parameter===')
    print(precision_recall_df.loc[precision_recall_df['f1 score'].idxmax()])

if __name__ == "__main__":
  run()