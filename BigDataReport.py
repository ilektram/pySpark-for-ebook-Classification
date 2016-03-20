import os
import sys
import re
import xml.etree.ElementTree as ET
import pickle

from math import log
from operator import add
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import LogisticRegressionWithLBFGS as LR

# Turn on 10-Cross Validation (set to True)
cross_validation = False


# Define functions
def header(matched, x):
    """
    Extracts book ID from header
    Params:
    matched = output of main_text function
    x = main_text function input
    Returns Key-Value pair tuples:
    Key = book ID as str or None if no book ID is found
    Value = main text as str
    """
    main_text = (matched.group(2))
    pattern = re.compile(r".*?E*?(book|text)\s#(\d+)", flags=(re.I | re.S))
    bookID = pattern.match(matched.group(1))
    if bookID is not None:
        return (str(bookID.group(2)), main_text)
    else:
        return (None, main_text)


def main_text(x):
    """
    Extracts main_text from book
    Params:
    x = RDD with key value pairs of the form (filename, book text)
    Returns header(book main text as str, x)
    """
    pattern = re.compile(r"(.*?)\*{3}.+?\*{3}(.*?)\*{3}.+\*{3}.*", flags=re.S)
    matched = pattern.match(x[1])
    if matched:
        return header(matched, x)
    else:
        return (None, None)


def splitFileWords(x):
    """ Splits text into words
    :param x: (key, whole text)
    :type x: tuple (str, str)
    :returns: A list of non empty words from the whole text
              [(key, word1), (key, word2), ... ]
    :rtype: list of tuples
    """
    fwLst = []
    wLst = re.split('\W+', x[1])
    for w in wLst:
        if w != '':
            fwLst.append((x[0], w))
    return fwLst


def remPlural(word):
    """ Formats a word to be in singular form. It achieves this by dropping the
    final s if there is one
    :param word: word to be processed
    :type word: string like
    :returns: word in singular form
    :rtype: str same as input
    """
    word = word.lower()
    if word.endswith('s'):
        return word[:-1]
    else:
        return word


def stopwords():
    """
    Creates list of english stopwords from lab3 file
    """
    # Read text files with stopwords as RDD and split by commas
    stopwordsRDD = (sc
                    .textFile('/data/store/labs/lab3/stopwords_en.txt')
                    .flatMap(lambda x: x.split(',')))
    stopwords = stopwordsRDD.collect()
    return stopwords


def b_hashVector(b_wcl, vsize):
    """
    Hashes wordcount vector
    Input: RDD, vector size
    Returns book ID, vector
    """
    b, wcl = b_wcl
    vec = [0] * vsize
    for wc in wcl:
        i = hash(wc[0]) % vsize
        vec[i] = vec[i] + wc[1]
    return (b, vec)


def to_namespace(str_in):
    """
    Turns strings to namespaces while parsing xml files
    Input: string of the form 'pgterms:ebook'
    Returns a string of the form "{namespace}string_value"
    """
    # Save namaspaces as dictionary
    namespaces = {
        "pgterms": "http://www.gutenberg.org/2009/pgterms/",
        "dcterms": "http://purl.org/dc/terms/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dcam": "http://purl.org/dc/dcam/"
    }

    key, value = str_in.split(':')
    return '{' + namespaces[key] + '}' + value


def parseXML(file_path):
    """
    Parses XML files to get tree root
    Input: XML file directory
    Returns root if parsing was successful and None if not
    """
    tree = ET.parse(file_path)
    try:
        root = tree.getroot()
    except ET.ParseError:
        root = None
    return root


def getID(root):
    """
    Parses XML to find eBook ID value
    Input: root of tree as given by parseXML() function
    Returns eBook ID as str
    """
    Node = root.find(to_namespace('pgterms:ebook'))
    if Node is None:
        return ''
    else:
        return Node.get(to_namespace('rdf:about'))


def getSubject(root):
    """
    Parses XML to find eBook subject values
    Input: root of tree as given by parseXML() function
    Returns list of all subjects for an ebook as strs
    """
    ret_list = []
    ebook = root.find(to_namespace('pgterms:ebook'))
    if ebook is None:
        return ret_list
    for node in ebook.findall(to_namespace('dcterms:subject')):
        desc = node.find(to_namespace('rdf:Description'))
        if desc is None:
            continue
        member_of = desc.find(to_namespace('dcam:memberOf'))
        if member_of.get(to_namespace('rdf:resource')) == "http://purl.org/dc/terms/LCSH":
            value = desc.find(to_namespace('rdf:value'))
            ret_list.append(value.text)
    return ret_list


def meta_id(x):
    """
    Matches and extracts the ID of the eBook as str
    Input: the output of getID() function
    Returns book ID as str or None if no book ID is found
    """
    pattern = re.compile(r".*?(\d+).*", flags=re.S)
    matched = pattern.match(x)
    if matched:
        return matched.group(1)
    else:
        return None


def get_subject_tuple(file_path):
    """
    Creates a tuple containing the ebook ID and its subjects
    Input: XML file directory
    Returns (ebook ID, [Subject strs]) or None if either field is empty
    """
    root = parseXML(file_path)
    if root is None:
        return None
    file_id = meta_id(getID(root))
    if not file_id:
        return None
    subject_list = getSubject(root)
    if subject_list is None or len(subject_list) == 0:
        return None
    return (file_id, subject_list)


def subject_Vector(bookSubjectRDD, hashedVectorRDD):
    """
    Creates a tuple containing the ebook subject and its hashed vector
    Each ebook may result in multiple tuples, due to more than 1 subjects
    Input: 2 RDDs of tuples with ID, subject list/hashed vector
    Returns RDD with tuples (subject, hashed vector)
    """
    # bookSubjectRDD : (book, [subject1, subject2 ...])
    # hashedVectorRDD: (book, [tfidfs, ...])
    subjectVectorRDD = (bookSubjectRDD
                        .join(hashedVectorRDD)
                        .map(lambda x: x[1])
                        .flatMapValues(lambda x: x))
    return subjectVectorRDD


def evaluate(model, ds):
    """
    Evaluates machine learning models
    Input: model, RDD of data set to evaluate
    Returns percentage accuracy (true positive rate)
    """
    results = ds.map(lambda x: (model.predict(x.features), x.label))
    truePositives = results.filter(lambda x: x[0] == x[1])
    result = float(truePositives.count()) / results.count()
    truePositives.unpersist()
    results.unpersist()
    return result


def crossValidation(hyperparameters, classifier, dataRDD, nFolds):
    """
    Performs cross validation of NB or LR models to optimise hyperparameters
    Input: hyperparameters as dictionary, model, RDD of data set to evaluate, number of folds
    Returns optimum hyperparameter for the model
    """
    # randomly split the data into n equal sized folds
    dataFolds = dataRDD.randomSplit([1] * nFolds)
    # initialise a vector of results
    metrics = [0] * len(hyperparameters)
    # loop through hyperparameters
    for i, params in enumerate(hyperparameters):
        # loop through folds
        for j in range(nFolds):
            # get validation data
            validationData = dataFolds[j]
            # get training data (everything else)
            trainingData = sc.emptyRDD()
            for k in range(len(dataFolds)):
                if k != j:
                    trainingData = trainingData.union(dataFolds[k])
            # train the model with given parameters
            if classifier is NaiveBayes:
                model = classifier.train(trainingData, **params)
            elif classifier is LR:
                model = classifier.train(trainingData, numClasses=20, **params)
            # We no longer need these, mark them to be removed
            trainingData.unpersist()
            # validate the model
            metric = evaluate(model, validationData)
            validationData.unpersist()

            # track the performance
            metrics[i] = metrics[i] + metric
    # return the best parameters
    bestIndex = metrics.index(max(metrics))
    return hyperparameters[bestIndex]


def regularizationParam(classifier):
    """
    Returns optimum regularisation parameter for a model,
    depending on whether cross validation is used or not
    Input: machine learning model, either NB or LG model
    Returns regularisation parameter to be used
    """
    if classifier == LR:
        if not cross_validation:
            param = {'regParam': 0.3}
        else:
            param = crossValidation([{'regParam': 0.1},
                                     {'regParam': 0.3},
                                     {'regParam': 1.0},
                                     {'regParam': 3.0},
                                     {'regParam': 10.0}], classifier, valSet, 10)
    elif classifier == NaiveBayes:
        if not cross_validation:
                param = {'lambda_': 0.1}
        else:
            param = crossValidation([{'lambda_': 0.1},
                                     {'lambda_': 0.3},
                                     {'lambda_': 1.0},
                                     {'lambda_': 3.0},
                                     {'lambda_': 10.0}], classifier, valSet, 10)
    print("Optimum regularisation parameter for ", classifier, " :", param)
    return param


# If this is the main program
if __name__ == "__main__":
    # Connect to Spark
    sc = SparkContext(appName="Big Data Report 2015")

    # Part 1
    # Load and parse non-empty files into RDD
    rawDataRDD = sc.emptyRDD()
    for root, dir, files in os.walk('/data/store/gutenberg/text-part/'):
        rawDataRDD = (rawDataRDD.union(
            sc.wholeTextFiles(root).filter(lambda x: x[1] != ""))
        )
    # Count files
    fileNum = rawDataRDD.count()
    print('Number of files: ', fileNum, '\n')
    # Use regex to capture header & footer & bookID:
    booksRDD = rawDataRDD.map(main_text)
    # Subset only matched ebooks & flatmap
    matchedBooksRDD = booksRDD.filter(lambda x: x[0] is not None)
    print('Number of matched books: ', matchedBooksRDD.count(), "\n")
    wordsRDD = matchedBooksRDD.flatMap(splitFileWords)
    # Read text files with stopwords as RDD and split by commas
    # Remove the empty string from words and remove stopwords and word "i"
    stopwords_list = stopwords()
    countWordsRDD = (wordsRDD
                     .map(lambda x: ((x[0], remPlural(x[1])), 1))
                     .filter(lambda x: len(x[0][1]) > 0)
                     .filter(lambda x: x[0][1] not in stopwords_list)
                     .filter(lambda x: x[0][1] is not 'i'))
    # Reduce by book to get total wordcounts
    bookWordCountRDD = (countWordsRDD
                        .reduceByKey(add)
                        .map(lambda x: (x[0][0], (x[0][1], x[1])))
                        .cache())
    # Rearrange tuples & reduce by word to get total wordcounts & save to disk
    wordBookCountRDD = (bookWordCountRDD
                        .map(lambda x: (x[1][0], [(x[0], x[1][1])]))
                        .reduceByKey(add)
                        .cache())
    # Calculate df for each word & save to disk
    dfRDD = (wordBookCountRDD
             .map(lambda x: (x[0], len(x[1]), x[1]))
             .cache())
    # Calculate total number of books
    TotalBooks = matchedBooksRDD.count()
    # Calculate IDF & save to disk
    idfRDD = dfRDD.map(lambda x: (x[0], log(TotalBooks/x[1]))).cache()
    # Calculate max word freq per file
    bookMaxFreqRDD = (bookWordCountRDD
                      .mapValues(lambda x: x[1])
                      .reduceByKey(max))
    wordMaxFreqRDD = bookWordCountRDD.join(bookMaxFreqRDD)
    # Calculate TF
    tfRDD = wordMaxFreqRDD.map(lambda x: (x[1][0][0], (x[0], x[1][0][1]/x[1][1])))
    # Calculate TF.IDF for each word per ebook and save to disk
    TfIdfRDD = (tfRDD
                .join(idfRDD)
                .map(lambda x: (x[1][0][0], [(x[0], x[1][0][1] * x[1][1])]))
                .reduceByKey(add)
                .cache())
    TfIdf = TfIdfRDD.collect()

    # Remove unneeded RDDs
    TfIdfRDD.unpersist()
    wordMaxFreqRDD.unpersist()
    bookMaxFreqRDD.unpersist()
    tfRDD.unpersist()
    idfRDD.unpersist()
    dfRDD.unpersist()
    countWordsRDD.unpersist()
    matchedBooksRDD.unpersist()
    wordsRDD.unpersist()
    booksRDD.unpersist()
    rawDataRDD.unpersist()

    # Save as pickle file
    with open("TfIdf.p", "wb") as f:
        pickle.dump(TfIdf, f)
    # Load pickle file
    with open("TfIdf.p", "rb") as f:
        TfIdf = pickle.load(f)
    # Bring to spark context
    TfIdfRDD = sc.parallelize(TfIdf)
    print("TfIdfRDD count: ", TfIdfRDD.count())
    # Use hashing trick
    hashSize = 10000
    hashVecRDD = TfIdfRDD.map(lambda x: b_hashVector(x, hashSize))

    # Part 2
    # Parse XML metadata files

    # Parse all files and extract book IDs and subjects
    # Save book IDs and subjects as tuples in a list
    bookSubjectList = []
    for root, dir, files in os.walk('/data/store/gutenberg/meta/'):
        for f in files:
            i = get_subject_tuple(os.path.join(root, f))
            if i is not None:
                bookSubjectList.append(i)
    # Create RDD for book IDs and subjects
    bookSubjectRDD = sc.parallelize(bookSubjectList)
    print("bookSubjectRDD count: ", bookSubjectRDD.count(), "\n")

    # Part 3
    # Re-arrange book subject RDD
    flatBookSubjectRDD = bookSubjectRDD.flatMapValues(lambda x: x)
    # Count subject occurencies and sort by descending order
    countSubjectRDD = (flatBookSubjectRDD
                       .map(lambda x: (x[1], 1))
                       .reduceByKey(add)
                       .sortBy(lambda x: x[1], ascending=False))
    # Find 20 most frequent ebook subjects
    topSubjects = countSubjectRDD.keys().take(20)
    print("20 Most Frequent Subjects in project Gutenberg: ", topSubjects, "\n")
    # Create labelled points according to subject
    subjectVecRDD = subject_Vector(bookSubjectRDD, hashVecRDD)
    print("subjectVecRDD length: ", subjectVecRDD.count(), "\n")
    # Create float IDs for each of the top 20 subjects
    subjectIdRDD = sc.parallelize(enumerate(topSubjects)).map(lambda x: (x[1], x[0]))
    # Keep only top 20 subjects' Vectors
    # Replace subject string by subject id
    topSubjectVecRDD = (subjectVecRDD
                        .join(subjectIdRDD)
                        .map(lambda x: (x[1][1], x[1][0])))
    # Create labeled points
    lblPntRDD = topSubjectVecRDD.map(lambda x_l: LabeledPoint(x_l[0], x_l[1]))
    # Split the data in testing, validation and training sets if cross_validation
    if cross_validation:
        trainSet, testSet, valSet = lblPntRDD.randomSplit([0.7, 0.1, 0.2], seed=0)
        print("Validation set size: ", valSet.count())
    # Or only in training and testing sets if not cross_validation
    elif not cross_validation:
        trainSet, testSet = lblPntRDD.randomSplit([0.9, 0.1], seed=0)
    print("Training set size: ", trainSet.count())
    print("Testing set size: ", testSet.count(), "\n")
    # Optimize NB reguralization parameter with 5-Fold Cross Validation
    nbParam = regularizationParam(NaiveBayes)
    # Run Naive Bayes model on training set
    nbModel = NaiveBayes.train(trainSet, **nbParam)
    # Evaluate Naive Bayes model on both training and testing sets
    print("Training accuracy of Naive Bayes: ")
    print(evaluate(nbModel, trainSet))
    print("Testing accuracy of Naive Bayes: ")
    print(evaluate(nbModel, testSet), "\n")
    numClasses = 20
    # Optimize LR reguralization parameter with 10-Fold Cross Validation
    logRegParam = regularizationParam(LR)
    # Run Logistic Regression model on training set
    logRegModel = LR.train(trainSet, numClasses=numClasses, **logRegParam)
    # Evaluate Logistic Regression model on both training and testing sets
    print("Training accuracy of Logistic Regression: ")
    print(evaluate(logRegModel, trainSet))
    print("Testing accuracy of Logistic Regression: ")
    print(evaluate(logRegModel, testSet), "\n")
    # Disconnect from Spark
    sc.stop()
