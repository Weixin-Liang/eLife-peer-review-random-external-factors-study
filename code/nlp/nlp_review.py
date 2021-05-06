import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import nltk
import random
import os
from collections import Counter, defaultdict
import re
import json 
import math
import matplotlib.pyplot as plt
import time
import csv
import pickle
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import os.path
from scipy.stats.stats import pearsonr, spearmanr
import scipy
import sklearn
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import utils


def load_nlp_review_data(DATA_ROOT='./nlp/'):

    ##################################
    # This two file jointly provide the full information. 
    ##################################
    df = pd.read_csv(DATA_ROOT + "tolabel.csv", sep="|")
    df = df[["Manuscript no.", "Reviewer ID", "CleanedComments", "Rec", "Suitable", "ShouldBe", "HumanLabel"]]
    df = df.set_index(["Manuscript no."])

    scored_bert = pd.read_csv(DATA_ROOT + "scored_reviews.csv", 
                            sep="\t", names=["id", "score", "dummy", "text"])
    df["score"] = list(scored_bert.score)
    df["Text"] = list(scored_bert.text)
    print('BERT score mean {:.3f}, std {:.3f}'.format(np.mean(scored_bert.score), np.std(scored_bert.score)) )
    df.drop(columns=["Rec", "Suitable", "ShouldBe",], inplace=True)
    print('df')
    print(df)

    # read in paper history stuff
    e = pd.read_csv(DATA_ROOT + "../eLife_Paper_history_2019_03_15.csv")
    e["Manuscript no."] = e["ms"]
    e = e.set_index(["Manuscript no."])
    e = e.dropna(subset=["full_decision"])

    # to get finaldecision, take last non-NA decision of the ones listed here
    # note that this excludes rejected by initial decision
    e["FinalDecision"] = e.apply(lambda x: list(x[["full_decision", "rev1_decision", "rev2_decision", "rev3_decision", "rev4_decision"]].dropna())[-1], axis=1)
    e["outcome"] = np.where(e["FinalDecision"] == "Accept Full Submission", 1, 0)
    df_e = df.join(e)

    print('df_e', df_e)
    analyze_review_outcome_score_alignment(df_e)
    return df_e


def analyze_review_outcome_score_alignment(df_e):

    ##################################
    # TODO: add a function here
    # RE’s decision compares with the reviewers’ sentiments. Can we do a quick correlation analysis of this?
    # Avg reviewer score vs. acceptance decision? Or min, max? 
    ##################################
    # build a dict: Manuscript no. --> final decision outcome (0.0/1.0), BERT scores
    manuscript_ID_to_reviews = defaultdict(list)
    manuscript_ID_to_outcome = dict()
    for i, line in tqdm(df_e.iterrows(), total=df_e.shape[0]):
        if math.isnan(line['ms']):
            continue # skip this row, problematic 
        manuscript_ID = int(line['ms']) # Manuscript no. 

        if line['FinalDecision'] == 'Accept Full Submission':
            manuscript_ID_to_outcome[manuscript_ID] = 1 # Accept Full Submission 1.0 otherwise 0.0
        elif line['FinalDecision'] == 'Reject Full Submission':
            manuscript_ID_to_outcome[manuscript_ID] = 0
        else:
            # review not finished, skipped. 
            continue

        existing_reviewer_IDs = [ tup[1] for tup in manuscript_ID_to_reviews[manuscript_ID]]
        if line['Reviewer ID'] not in existing_reviewer_IDs: 
            review_BERT_score =  line['score']
        manuscript_ID_to_reviews[manuscript_ID].append( [ review_BERT_score, int(line['Reviewer ID']), ] ) 

    ##################################
    # Explore avg, min, max
    ##################################
    manuscript_ID_to_avg_score = dict()
    manuscript_ID_to_min_score = dict()
    manuscript_ID_to_max_score = dict()
    for manuscript_ID, review_list in manuscript_ID_to_reviews.items():
        review_list.sort(key=lambda x: x[0]) # will sort only based on scores 
        score_list = [tup[0] for tup in review_list]

        avg_score = np.mean(score_list)
        min_score = score_list[0]
        max_score = score_list[-1]

        manuscript_ID_to_avg_score[manuscript_ID] = avg_score
        manuscript_ID_to_min_score[manuscript_ID] = min_score
        manuscript_ID_to_max_score[manuscript_ID] = max_score
    
    def calculate_correlation(manuscript_ID_to_score, name):
        # final output: 
        # A. numerical output: two bins, acc, rej: avg and std of scores. for avg score, max score, min score; or correlation numbers. auc score? 
        # B. graphical output: histogram
        outcome_array = []
        score_accray = []
        for manuscript_ID in sorted(manuscript_ID_to_reviews.keys()):
            outcome_array.append(manuscript_ID_to_outcome[manuscript_ID])
            score_accray.append(manuscript_ID_to_score[manuscript_ID])
        correlation, p_value = scipy.stats.pointbiserialr(outcome_array, score_accray)
        auc = sklearn.metrics.roc_auc_score(outcome_array, score_accray)
        print('Aggregation method: {}'.format(name))
        print('Point biserial correlation coefficient: {:.3f}  and its p-value:'.format(correlation), p_value)
        print('AUC: {:.3f}'.format(auc))
        return 

    calculate_correlation(manuscript_ID_to_score = manuscript_ID_to_avg_score, name='average')   
    calculate_correlation(manuscript_ID_to_score = manuscript_ID_to_min_score, name='min')   
    calculate_correlation(manuscript_ID_to_score = manuscript_ID_to_max_score, name='max')   

    return 
    

def load_reviewer_data(DATA_ROOT='./nlp/'):

    reviewers = pd.read_csv(DATA_ROOT + "gender_reviewers.csv", error_bad_lines=False)
    # this is wrong
    reviewers_data = pd.DataFrame(reviewers.groupby("Reviewer ID")["Reviewer name"].count())
    reviewers_data.columns = ["reviewer_count"]

    reviewers["review_count"] = reviewers.groupby("Reviewer ID")["gender"].transform("count")

    print('reviewers')
    print(reviewers)

    fuzzy_matcher = utils.Insitution_Fuzzy_Mather()

    reviewer_info_dict = dict()


    for i, line in tqdm(reviewers.iterrows(), total=reviewers.shape[0]):
        this_race = 'N/A'
        match_name, alpha_two_code = fuzzy_matcher.match(email=line['Reviewer email'], institution_name=line['Reviewer institution'])

        reviewer_info_dict[line['Reviewer ID']] = {
            'people_ID': int(line['Reviewer ID']),
            'name': line['Reviewer name'],
            'institution': line['Reviewer institution'], 
            'email': line['Reviewer email'], 
            'race': this_race, 
            'country_code': alpha_two_code, # -2 
            'gender': line['gender'], # -1 
        }

    return reviewer_info_dict





if __name__ == '__main__':

    df = load_nlp_review_data(DATA_ROOT='./')
    for i, line in tqdm(df.iterrows(), total=df.shape[0]):
        # pass 
        print('line', line)
        if i > 5:
            break
        pass 

    load_reviewer_data(DATA_ROOT='./')
