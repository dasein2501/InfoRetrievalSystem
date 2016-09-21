#!/usr/bin/python
from __future__ import division
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models, similarities
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import re
import getopt
import sys


def preprocess_document(doc):
  stopset = set(stopwords.words('english'))
  stemmer = PorterStemmer()
  tokens = wordpunct_tokenize(doc)
  clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
  final = [stemmer.stem(word) for word in clean]
  return final


def get_relevant_documents(f, query=None):
  """Get relevant documents for the provided query or all docs.
  @param String f: relative path to file.
  @param String query: query for which relevant docs are returned.
  @return Set relevant: set of relevant documents.
  """
  relevant = np.array([line.split()[::2] for line in open(f)])
  return set(relevant[relevant[:, 0] == query, 1]) if query else set(relevant[:, 1])


def read_file(filepath):
  """Read file with given format.
  @param String filepath: path to file.
  @return Generator[Tuple]: [(id,doc)].
  """
  doc = ""
  doc_id = 0
  file_format = "\.I\s(?P<doc_id>\d+)|(?!\.W)(?P<doc>.+)"
  pattern = re.compile(file_format)
  with open(filepath, 'r') as f:
    for line in f:
      m = pattern.match(line)
      if m and m.group("doc_id"):
        if doc:
          yield tuple([doc_id, doc])
          doc = ""
        doc_id = int(m.group("doc_id"))
      elif m and m.group("doc"):
        doc += m.group("doc").strip('\n\r ')
  yield tuple([doc_id, doc])


def create_dictionary(corpus):
  """Create dictionary and store it.
  @param [String] corpus: collection of documents.
  @return {dict} dictionary: dictionary of (word:id).
  """
  pdocs = [preprocess_document(doc[1]) for doc in corpus]
  dictionary = corpora.Dictionary(pdocs)
  dictionary.save('/tmp/irs.dict')
  return dictionary


def docs2bows(corpus, dictionary):
  """Create a bag-of-words for the corpus.
  @param [String] corpus: collection of documents.
  @param {dict} dictionary: dictionary of tokens.
  @return [[Tuple]] vectors: collection of bow for each doc.
  """
  pdocs = [preprocess_document(d[1]) for d in corpus]
  vectors = [dictionary.doc2bow(doc) for doc in pdocs]
  return vectors


def create_boolean_model(corpus):
  dictionary = create_dictionary(corpus)
  bow = docs2bows(corpus, dictionary)
  # Boolean model
  boolean = [[(w[0], 1) for w in v] for v in bow]
  corpora.MmCorpus.serialize('/tmp/irs_docs.mm', boolean)
  # Index against TF model
  index = similarities.MatrixSimilarity(boolean, num_features=len(dictionary))
  return boolean, dictionary, index


def create_tf_model(corpus):
  dictionary = create_dictionary(corpus)
  tf = docs2bows(corpus, dictionary)
  # TF model
  tf = [[(w[0], 1 + np.log2(w[1])) for w in v] for v in tf]
  corpora.MmCorpus.serialize('/tmp/irs_docs.mm', tf)
  # Index against TF model
  index = similarities.MatrixSimilarity(tf, num_features=len(dictionary))
  return tf, dictionary, index


def create_tf_idf_model(corpus):
  dictionary = create_dictionary(corpus)
  bow = docs2bows(corpus, dictionary)
  tfidf = models.TfidfModel(bow)
  corpora.MmCorpus.serialize('/tmp/irs_docs.mm', bow)
  index = similarities.MatrixSimilarity(bow, num_features=len(dictionary))
  return tfidf, dictionary, index


def launch_query(corpus, q, model_type, model, dictionary, index, top):
  """Execute query.
  @param Int top: Show only top retrievals.
  @param Object index: Matrix similarities
  @param Dict dictionary: Corpus dictionary
  @param Object model: Info. retrieval model
  @param String model_type: Type of model
  @param String corpus: path to corpus.
  @param Tuple(int,String) q: query_id and query.
  """
  pq = preprocess_document(q[1])
  vq = dictionary.doc2bow(pq)
  relevant_docs = get_relevant_documents("MED.REL", str(q[0]))
  n_retrieved = 0
  n_relevant = 0
  precision = np.array([])
  recall = np.array([])

  if model_type == "tf":
    q = [(w[0], 1 + np.log2(w[1])) for w in vq]
  elif model_type == "tfidf":
    q = model[vq]
  elif model_type == "boolean":
    q = [(w[0], 1) for w in vq]

  sim = index[q]
  ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)

  for doc, score in ranking[:top]:
    if str(corpus[doc][0]) in relevant_docs:
        n_relevant += 1
    n_retrieved += 1
    precision = np.append(precision, n_relevant / n_retrieved)
    recall = np.append(recall, n_relevant / len(relevant_docs))
    print "[ Score = " + "%.3f" % round(score, 3) + " | ID: " + str(corpus[doc][0]) + " ]\n" + corpus[doc][
     1] + "\n"
  return recall, precision


def interpolate_precision(recall, precision, std_recall_points):
  """
  Interpolated precision at given points.
  @param Collection recall: Recall at relevant points
  @param Collection precision: Precision at relevant points
  @param Collection std_recall_points: Standard recall points
  @return Collection interpolated precision.
  """
  return np.array(map(lambda x: max(precision[recall >= x]), std_recall_points))

def p_r_graph(std_recall_points):
  """
  Precision-Recall curve at recall points.
  @param Collection std_recall_points.
  @return Plot Object
  """
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  plt.xticks(std_recall_points)
  ax1.set_title("Precision vs Recall")
  ax1.set_xlabel('Recall')
  ax1.set_ylabel('Precision')
  return ax1

def usage():
  print '\nUsage: '
  print (
    sys.argv[0] + ' [-h,--help] --model=[boolean,tf,tfidf,all] ' +
    '--corpus-file=<file1>' +
    ' --queries-file=<file2> --top=rank_entries ' +
    ' [offsetQuery] [finalQuery]\n'
  )

def main():
  model_type = qfile = cfile = offset = end = top = index = model = dictionary = shape = None
  current_models = []
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
    elif o == "--model":
      model_type = a
    elif o == "--queries-file":
      qfile = a
    elif o == "--corpus-file":
      cfile = a
    elif o == "--top":
      top = int(a)
    else:
      assert False, "unhandled option"

  if not qfile:
    queries = [('', raw_input('Enter your query: '))]
  else:
    queries = list(read_file(qfile))

  corpus = list(read_file(cfile))

  if model_type == "tf":
    current_models = ["tf"]
  elif model_type == "tfidf":
    current_models = ["tfidf"]
  elif model_type == "boolean":
    current_models = ["boolean"]
  elif model_type == "all":
    current_models = ["tf", "tfidf", "boolean"]

  if len(args) == 2:
    offset = int(args[0])
    end = int(args[1])
  elif len(args) == 1:
    offset = int(args[0])
    end = offset + 1

  std_recall_points = np.arange(.0, 1.1, .1)
  g = p_r_graph(std_recall_points)

  for m in current_models:
    if m == "tf":
      model, dictionary, index = create_tf_model(corpus)
      shape = 'r-o'
    elif m == "tfidf":
      model, dictionary, index = create_tf_idf_model(corpus)
      shape = 'b-o'
    elif m == "boolean":
      model, dictionary, index = create_boolean_model(corpus)
      shape = 'g-o'

    inter_precision = 0
    for q in queries[offset:end]:
      print "\nResults of query " + str(q[0]) + ": " + q[1] + "\n"
      recall, precision = launch_query(corpus, q, m, model, dictionary, index, top)
      inter_precision += interpolate_precision(recall, precision, std_recall_points)

    avg_precision = inter_precision / len(queries[offset:end])
    g.plot(std_recall_points, avg_precision, shape, label=m)

  if qfile:
      g.legend()
      plt.show()

if __name__ == "__main__":
  main()
