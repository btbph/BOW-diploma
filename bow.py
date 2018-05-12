import cv2
import pandas as pd
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split

# добавил метод опорных векторов


class OpenCVBow:
    dictionary_size = 0
    dictionary = {}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    bow_trainer = {}
    bow_extractor = {}
    voc = {}
    svm = {}
    sift = cv2.xfeatures2d.SIFT_create()

    def __init__(self, d_s):
        self.dictionary_size = d_s
        self.bow_trainer = cv2.BOWKMeansTrainer(self.dictionary_size)
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.sift, cv2.BFMatcher(cv2.NORM_L2))
        '''
            best result 
            gamma = 50
            c = default
            res = 11%
        '''
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setGamma(50)
        # self.svm.setC(10)
        self.svm.setKernel(cv2.ml.SVM_RBF)

    def extract_sift(self, fn):
        im = cv2.imread(fn, 0)
        # im = cv2.resize(im, (300, 300))
        return self.sift.compute(im, self.sift.detect(im))[1]

    def bow_features(self, img):
        return self.bow_extractor.compute(img, self.sift.detect(img))

    def train_bow_extractor(self, df, path_dataset):
        print('start to form bow_extractor ' + time.ctime())
        for index, row in df.iterrows():
            img_path = os.path.join(path_dataset, row[0])
            self.bow_trainer.add(self.extract_sift(img_path))
        self.voc = self.bow_trainer.cluster()
        self.bow_extractor.setVocabulary(self.voc)
        fs = cv2.FileStorage('vocabulary.yml', cv2.FILE_STORAGE_WRITE)
        fs.write('voc', self.voc)
        fs.release()
        print('end to form bow_extractor ' + time.ctime())

    def train_data_for_svm(self, df, path_dataset):
        print('start to create train data for svm ' + time.ctime())
        train_data, train_labels = [], []
        for index, row in df.iterrows():
            img_path = os.path.join(path_dataset, row[0])
            tmp_img = cv2.imread(img_path, 0)
            # tmp_img = cv2.resize(tmp_img, (300, 300))
            train_data.extend(self.bow_features(tmp_img))
            train_labels.append(self.logo_number[row[1]])
        fs = cv2.FileStorage('train_data.yml', cv2.FILE_STORAGE_WRITE)
        fs.write('train_data', np.array(train_data))
        fs.write('train_labels', np.array(train_labels))
        fs.release()
        print('finish creating test data from svm ' + time.ctime())

    def train(self):
        # bow part
        df = pd.read_csv('./flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt', sep=" ", header=None)
        path_dataset = './flickr_logos_27_dataset/dataset'
        train, test = train_test_split(df, random_state=42)

        print('form dict for coding lables ' + time.ctime())
        logo_arr = pd.unique(df.iloc[:, 1])
        self.number_logo = {index: logo for (index, logo) in enumerate(logo_arr)}
        self.logo_number = {logo: index for (index, logo) in enumerate(logo_arr)}
        print('finish creating dict ' + time.ctime())

        self.train_bow_extractor(train, path_dataset)
        fs = cv2.FileStorage('vocabulary.yml', cv2.FILE_STORAGE_READ)
        res = fs.getNode('voc').mat()
        self.bow_extractor.setVocabulary(res)

        self.train_data_for_svm(train, path_dataset)
        fs = cv2.FileStorage('train_data', cv2.FILE_STORAGE_READ)
        train_data = fs.getNode('train_data').mat()
        train_labels = fs.getNode('train_labels').mat()

        # svm part
        print('start to train svm ' + time.ctime())
        self.svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_labels))
        print('finish training svm ' + time.ctime())

        self.test_predict(test, path_dataset)

    def predict(self, path_img):
        bf = self.bow_features(cv2.imread(path_img, 0))
        _, res = self.svm.predict(bf)
        # a, res1 = self.svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT | cv2.ml.STAT_MODEL_UPDATE_MODEL)
        logo = self.number_logo[int(res[0][0])]
        print(logo)

    def test_predict(self, test, path_dataset):
        good = 0
        total = 0
        print('start train predict ' + time.ctime())
        for index, row in test.iterrows():
            img_path = os.path.join(path_dataset, row[0])
            correct_class = self.logo_number[row[1]]
            tmp_img = cv2.imread(img_path, 0)
            # tmp_img = cv2.resize(tmp_img, (300, 300))
            img_features = self.bow_features(tmp_img)
            _, res = self.svm.predict(img_features)
            if correct_class == int(res[0][0]):
                good = good + 1
            total = total + 1
        print('finish train predict ' + time.ctime())
        print(str(100*good/total)+'%')
