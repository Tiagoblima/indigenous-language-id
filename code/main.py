import os
import sys
import time

from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC

from WordRank.wordrank import *
from params import CLASSES_
from textcat import TextCat
import nltk

nltk.download('punkt')

profiles = '../profiles/'
datasets = '../data'


# textcat = TextCat()


# textcat = TextCat()


def save_profile():
    os.mkdir('profiles')
    for dataset in os.listdir(datasets):

        path = os.path.join(datasets, dataset)
        out_path = os.path.join(profiles, dataset)
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
        print(dataset)
        for t in range(1, 6):
            val_dir = os.path.join(out_path, 'val_' + str(t))
            os.mkdir(val_dir)

            train_df = pd.read_csv(os.path.join(path, 'train_{}.csv'.format(t)))
            for lang in np.unique(train_df['LANG']):
                lang_txt = train_df[lang == train_df['LANG']]['TEXT'].tolist()
                dicta = textcat.profile(' '.join(lang_txt))
                for key in dicta:
                    open(os.path.join(val_dir, lang + '.txt'), 'a', encoding='utf-8').write(
                        key + ' ' + str(dicta[key]) + '\n')


def get_report(y_pred, y_, overall_metric='weighted avg'):
    report = classification_report(y_, y_pred,
                                   target_names=CLASSES_,
                                   output_dict=True)
    report = pd.DataFrame.from_dict(report).T.drop(columns=['support'])
    result_dict = report.loc[['macro avg', 'weighted avg']].loc[overall_metric].to_dict()

    return result_dict


# save_profile()
# breakpoint()
exps_result = []
for dataset in os.listdir(datasets):
    path = os.path.join(datasets, dataset)
    out_path = os.path.join(profiles, dataset)
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    for i in range(1, 6):
        wordrank = WordRankClassifier(return_names=True)

        train_file = 'train_{}.csv'.format(i)
        sys.stdout.write('\rRunning dataset {} Validation {}'.format(dataset, i))
        sys.stdout.flush()

        if os.path.isfile(path):
            continue

        train_df = pd.read_csv(os.path.join(path, train_file))
        y_true = train_df['LANG'].to_numpy()
        X_train = train_df['TEXT'].to_numpy()

        test_df = pd.read_csv(os.path.join(datasets, 'test.csv'))
        y_test = test_df['LANG'].to_numpy()
        X_test = test_df['TEXT'].to_numpy()

        ## Training WordRank ############

        wordrank.fit(X_train, y_true)

        start = time.time()
        y_pred = wordrank.predict(X_test)

        pred_time = time.time() - start
        report = get_report(y_pred, y_test)
        report.update({'dataset_name': dataset, 'pred_time': pred_time,
                       'classifier_name': 'WordRank'})
        exps_result.append(report)
        ##### TextCat ###############

        profiles = '../profiles/' + dataset + '/val_' + str(i) + '/'
        textcat = TextCat(path_to_profiles=profiles)

        start = time.time()
        y_pred = list(map(textcat.guess_language, X_test))
        pred_time = time.time() - start
        report = get_report(y_pred, y_test)
        report.update({'dataset_name': dataset, 'pred_time': pred_time,
                       'classifier_name': 'TextCat'})

        exps_result.append(report)
        ###########################################

        #### Tranning ML models

        pipe = Pipeline([('count', CountVectorizer(analyzer='char', ngram_range=(2, 3))),
                            ('tfid', TfidfTransformer())]).fit(X_train)
        X_train = pipe.transform(X_train)

        X_test = pipe.transform(X_test)

        #### Linear SVC ######################
        linear_clf = LinearSVC()
        linear_clf.fit(X_train, y_true)
        start = time.time()
        y_pred = linear_clf.predict(X_test)
        pred_time = time.time() - start
        report = get_report(y_pred, y_test)
        report.update({'dataset_name': dataset, 'pred_time': pred_time,
                       'classifier_name': 'LinearSVC'})
        exps_result.append(report)
        ################ MultinomialNB ############################

        mnb_clf = MultinomialNB()
        mnb_clf.fit(X_train, y_true)

        start = time.time()
        y_pred = mnb_clf.predict(X_test)
        pred_time = time.time() - start
        report = get_report(y_pred, y_test)
        report.update({'dataset_name': dataset, 'pred_time': pred_time,
                       'classifier_name': 'MultinomialNB'})
        exps_result.append(report)
        ######################################
pd.DataFrame(exps_result).to_csv('../results.csv', index_label=False, index=False)
