# WordRank

## Introduction

WordRank is a language classification algorithm which uses n-grams frequency to
predict the language of given document. The algorithm follows the pipeline:

1. Calculate the frequency of each n-gram and select the 100 more re-currents in
the training set of each language. From now on it will be our training set.
2. Calculate the 100 most frequent words in the test set of a given language.
3. Verify with the 100 most frequent words of each document of the test set is in the training
   sets of each language.
4. The language with the training set with the highest number of present words
   in both test and training is the set of the language that we want to identify.

### Demo

    word_clf = WordRankClassifier()
    
    word_clf.fit(X_train, y_train)
    
    y_pred = word_clf.predict(X_test)
