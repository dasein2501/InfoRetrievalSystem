#!/usr/bin/python
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models, similarities
from operator import itemgetter
import numpy as np
import re
import getopt, sys

def preprocess_document(doc):
  stopset = set(stopwords.words('english'))
  stemmer = PorterStemmer()
  tokens = wordpunct_tokenize(doc)
  clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
  final = [stemmer.stem(word) for word in clean]
  return final

def getRelevantDocuments(f,query=None):
  """Get relevant documents for provided query or all docs.
  @param String f: relative path to file.
  @param String query: query for which relevant docs. are returned
  """
  relevant = np.array([line.split()[::2] for line in open("MED.REL")])
  return relevant[relevant[:,0] == query, 1] if query else relevant[:,1]

def read_file(file):
  """Read file with given format.
  @param String file: path to file.
  @return [String] corpus: collection of documents/queries.
  """
  #corpus = {}
  f = open(file)
  raw = f.read()
  corpus = filter(None,re.split(r'\.I\s[0-9]+\r\n\.W\r\n',raw))

  '''doc_id = 0
  doc = ""
  new_doc = False

  with open(file,"r") as f:
    for line in f:
      if re.search(r'\.I\s[0-9]+',line):# and re.search(r'\.W\r\n',line)
        doc_id = line.split()[1]
        new_doc = True
        pass
      elif not re.search(r'\.W\r\n',line) and new_doc:
        doc = doc + line
      else:'''
  return corpus

def create_dictionary(corpus):
  """Create dictionary and store it.
  @param [String] corpus: collection of documents.
  @return {dict} dictionary: dictionary of (word:id).
  """
  pdocs = [preprocess_document(doc) for doc in corpus]
  dictionary = corpora.Dictionary(pdocs)
  dictionary.save('/tmp/irs.dict')
  return dictionary

def docs2bows(corpus, dictionary):
  """Create a bag-of-words for the corpus and store it.
  @param [String] corpus: collection of documents.
  @param {dict} dictionary: dictionary of tokens.
  @return [[Tuple]] vectors: collection of bow for each doc.
  """
  pdocs = [preprocess_document(d) for d in corpus]
  vectors = [dictionary.doc2bow(doc) for doc in pdocs]
  return vectors

def create_TF_model(corpus):
  dictionary = create_dictionary(corpus)
  bow = docs2bows(corpus, dictionary)
  # TF model
  tf = [[(w[0], 1 + np.log2(w[1])) for w in v] for v in bow]
  corpora.MmCorpus.serialize('/tmp/irs_docs.mm', tf)
  # Index against TF model
  index = similarities.MatrixSimilarity(tf, num_features=len(dictionary))
  return tf, dictionary, index

def create_TF_IDF_model(corpus):
  dictionary = create_dictionary(corpus)
  bow = docs2bows(corpus, dictionary)
  tfidf = models.TfidfModel(bow)
  corpora.MmCorpus.serialize('/tmp/irs_docs.mm', bow)
  index = similarities.MatrixSimilarity(bow, num_features=len(dictionary))
  return tfidf, dictionary, index

def launch_query(corpus, q, model_type, model, dictionary, index, top):
  """Execute query.
  @param String corpus: path to corpus.
  @param String q: query.
  """
  pq = preprocess_document(q)
  vq = dictionary.doc2bow(pq)

  if model_type == "tf":
    q = [(w[0], 1 + np.log2(w[1])) for w in vq]
  elif model_type == "tfidf":
    q = model[vq]

  sim = index[q]
  ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
  for doc, score in ranking[:top]:
    print "[ Score = " + "%.3f" % round(score,3) + " | Doc_ID: "+str(doc+1)+" ]\n" + corpus[doc];

def usage():
  print '\nUsage: '
  print (
    sys.argv[0]+' [-h,--help] --model=[tf,tfidf] '+ 
    '--corpus-file=<file2>' + 
    ' --queries-file=<file1> --top=rank_entries '+
    ' [offsetQuery] [finalQuery]\n'
  )

def main():
  model_type = qfile = cfile = offset = end = top = None
  try:
    opts, args = getopt.getopt(sys.argv[1:], "h", [
      "help", "model=", "corpus-file=", "queries-file=", "top="])
  except getopt.GetoptError as err:
    print str(err)
    usage()
    sys.exit(2)
  if len(args) > 2 or not opts:
    print "\nError: Illegal arguments."
    usage()
    sys.exit()
  for o, a in opts:
    if o in ("-h", "--help"):
      usage()
      sys.exit()
    elif o =="--model":
      model_type = a
    elif o =="--queries-file":
      qfile = a
    elif o =="--corpus-file":
      cfile = a
    elif o =="--top":
      top = int(a)
    else:
        assert False, "unhandled option"
  queries = read_file(qfile)
  corpus = read_file(cfile)
  if model_type == "tf":
    model, dictionary, index = create_TF_model(corpus)
  elif model_type == "tfidf":
    model, dictionary, index = create_TF_IDF_model(corpus)
  if len(args) == 2:
    offset = int(args[0])
    end = int(args[1])
  elif len(args) == 1:
    offset = int(args[0])
    end = offset + 1
  for q in queries[offset:end]:
    print "\nResults of query " + str(queries.index(q)) + " : " + q
    launch_query(corpus,q,model_type,model,dictionary,index,top)

if __name__ == "__main__":
    main()

# (1) Allow user to provide feedback to improve the retrieval. Use
# Rocchio's method for relevance feedback. Read paper and implement.
# Precision = doc_relevant_retrieved / doc_retrieved
# Recall = doc_relevant retrieved / doc_relevant in the collection