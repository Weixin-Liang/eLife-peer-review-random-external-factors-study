import pandas as pd
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
from scipy.stats import chi2_contingency, chisquare
from utils import get_senior_meta_data, get_reviewing_editor_meta_data
from nlp.nlp_review import load_nlp_review_data, load_reviewer_data
import matplotlib as mpl
mpl.style.use('seaborn')
ROOT_DIR = './output/'

class People_Meta_Data:

    def __init__(self, people: str):

        ##################################
        # Functions to load meta data 
        ##################################
        call_function_map = {
            'senior editors': get_senior_meta_data,
            'reviewing editors': get_reviewing_editor_meta_data,
            'authors': None,
            'reviewers': load_reviewer_data, 
        }
        assert people in call_function_map

        self.people = people # str 
        pkl_cache_path = 'cache/' + people + '.pkl'
        
        # if os.path.isfile(pkl_cache_path):
        if True: # Force Read from Cache 
        # if False: # Force Re-create Cache 

            ##################################
            # Read from the cache   
            ##################################

            with open(pkl_cache_path, 'rb') as pkl_file:
                self.people_meta_data = pickle.load( pkl_file )
        else:

            load_fn = call_function_map[people]
            self.people_meta_data = load_fn()

            ##################################
            # Write to the cache   
            ##################################
            
            with open(pkl_cache_path, 'wb') as pkl_file:
                pickle.dump( self.people_meta_data, pkl_file )
        

        return 


    def query_field(self, people_ID: int, field: str):

        assert field in ['people_ID', 'name', 'institution', 'email', 'race', 'country_code', 'gender']
        meta_data_entry = self.people_meta_data[people_ID]
        data = meta_data_entry = [field]
        return data


    def query_all(self, people_ID: int):

        if people_ID not in self.people_meta_data:
            return None 
        meta_data_entry = self.people_meta_data[people_ID]
        return meta_data_entry




class AnchoringEffectCounter:
    def __init__(self, prefix_zeros_len, choice):
        assert choice in ['accept', 'reject']
        self.choice = choice 
        self.prefix_zeros_len = prefix_zeros_len

        self.accumulate_encourage_after_prefix = 0
        self.accumulate_reject_after_prefix = 0
        return 

    def update(self, bit_array):
        # bit_array is a sequence of int of 0, 1
        reject_after_prefix, encourage_after_prefix = 0, 0
        total_len = len(bit_array)
        for current_position in range( self.prefix_zeros_len, total_len): 
            # we only consider after successive rejects / accepts
            sum_prefix = sum(bit_array[current_position-self.prefix_zeros_len:current_position])
            target_value = 0 if self.choice == 'reject' else self.prefix_zeros_len
            if sum_prefix != target_value:
                continue  # does not match 
            if bit_array[current_position] == 1:
                encourage_after_prefix += 1
            else:
                reject_after_prefix +=1 
        ##################################
        # Book-keeping
        ##################################
        self.accumulate_encourage_after_prefix += encourage_after_prefix
        self.accumulate_reject_after_prefix += reject_after_prefix
        # interval = z * sqrt( (accuracy * (1 - accuracy)) / n)
        if encourage_after_prefix == 0 and reject_after_prefix == 0:
            condition_encourageRate = 'N/A'
        else:
            condition_encourageRate = encourage_after_prefix / (encourage_after_prefix + reject_after_prefix)
            condition_encourageRate_interval = 1.96 * math.sqrt( (condition_encourageRate * (1 - condition_encourageRate)) / (encourage_after_prefix + reject_after_prefix) )
        return condition_encourageRate

    def summary(self):
        # print weighted average 
        encourage_after_prefix = self.accumulate_encourage_after_prefix
        reject_after_prefix = self.accumulate_reject_after_prefix
        sum_sample_count = encourage_after_prefix+reject_after_prefix
        # print('summary', sum_sample_count)
        
        condition_encourageRate = encourage_after_prefix / sum_sample_count if sum_sample_count>0 else 0. 
        condition_encourageRate_interval = 1.96 * math.sqrt( (condition_encourageRate * (1 - condition_encourageRate)) / sum_sample_count ) if sum_sample_count>0 else 0. 
        # return self.prefix_zeros_len, condition_encourageRate, condition_encourageRate_interval
        return condition_encourageRate, condition_encourageRate_interval



class EntryAggregator:

    def __init__(self, aggregate_field : str, people : str):
        self.bins_dict = dict()
        self.aggregate_field = aggregate_field
        self.people = people
        return  

    def update(self, people_query_ret : dict, accept_score : float, date_time_obj, qc_date_time_obj=None):
        # Y axis: single, all, group by country / race / gender 

        # all meta field ['people_ID', 'name', 'institution', 'email', 'race', 'country_code', 'gender']

        if self.aggregate_field == 'all':
            aggregate_field_value = 'all' # aggregate all data into one bin 
        elif self.aggregate_field == 'weekday':
            aggregate_field_value = date_time_obj.weekday() # 0-6 

        elif self.aggregate_field == 'weekend':
            weekday = date_time_obj.weekday() # 0-6 
            if weekday == 5 or weekday == 6 : # Saturday, Sunday
                aggregate_field_value = 'weekends'
            else:
                aggregate_field_value = 'weekdays'

        elif self.aggregate_field == 'year_month':
            if date_time_obj.year == 2019:
                return # skip 2019
            aggregate_field_value = str(date_time_obj.month)  +  ' ' + str(date_time_obj.year)   # 1-12

        elif self.aggregate_field == 'month':
            aggregate_field_value = date_time_obj.month # 1-12

        elif self.aggregate_field == 'year':
            aggregate_field_value = date_time_obj.year

        elif self.aggregate_field == 'season':

            month = date_time_obj.month
            if month == 12 or month == 1 or month == 2:
                aggregate_field_value = '4_winter'
            elif month == 3 or month == 4 or month == 5:
                aggregate_field_value = '1_spring'
            elif month == 6 or month == 7 or month == 8:
                aggregate_field_value = '2_summer'
            elif month == 9 or month == 10 or month == 11:
                aggregate_field_value = '3_fall'

        elif self.aggregate_field == 'US':
            aggregate_field_value = 'US' if people_query_ret['country_code'] == 'US' else 'non-US'
        
        elif self.aggregate_field == 'race & gender':
            aggregate_field_value = people_query_ret['race'] + ' & ' + people_query_ret['gender']

        elif self.aggregate_field in ['people_ID', 'race', 'country_code', 'gender']: # 'institution', 

            ##################################
            # Can be fetched directly 
            ##################################
            assert self.aggregate_field in people_query_ret
            aggregate_field_value = people_query_ret[self.aggregate_field]


        ##################################
        # Collector general update code  
        ##################################
        if aggregate_field_value not in self.bins_dict:
            self.bins_dict[aggregate_field_value] = []

        ##################################
        # The time gap of decision made 
        ##################################
        delta_day = (date_time_obj - qc_date_time_obj).days # also add the time gap to approximate the time taken to made the decision 
        self.bins_dict[aggregate_field_value].append((date_time_obj, accept_score, delta_day))

        return 
    


    ##################################
    # Collect anchoring effect
    # show successive rejects/ accepts 
    ##################################
    def filter_people_ID(self):
        assert self.aggregate_field == 'people_ID'

        ##################################
        # Step 1: Do post-processing
        # sort, and filter - for anchoring analysis 
        ##################################
        people_ID_sort_list = []
        for bin_key in self.bins_dict.keys(): # people_ID 
            ##################################
            # Filter out people that handles too few submisions
            ##################################
            total_len = len(self.bins_dict[bin_key])
            if self.people == 'reviewing editors':
                if total_len < 20:
                    continue # skip this editor
            elif self.people == 'senior editors':
                if total_len < 100:
                    print('Skipped senior editors, too few submissions: ', bin_key, total_len)
                    continue # skip this editor
            elif self.people == 'reviewers':
                # do not filter here 
                pass 

            encourageRate = sum([ tup[1] for tup in self.bins_dict[bin_key]]) / total_len
            people_ID_sort_list.append( (encourageRate, bin_key) )

        people_ID_sort_list.sort() # sort based on the encourage rate 
        # people_ID_sort_list.sort(key=lambda x: x[1]) # sort based on the ID
        people_ID_sorted_keys = [ t[1] for t in people_ID_sort_list]

        return people_ID_sorted_keys


    def summary(self):

        if self.aggregate_field == 'people_ID':

            ##################################
            # Centralized Sorting 
            ##################################
            for people_ID in sorted(self.bins_dict.keys()):
                total_len = len(self.bins_dict[people_ID])
                self.bins_dict[people_ID].sort(key=lambda x: x[0]) # sort based on date, need to be explicit 
                # self.bins_dict[people_ID].sort() # sort based on date, need to be explicit 


            ##################################
            # Additional Analysis: Decision Time Gap Study 
            ##################################
            print('Decision Time Gap')
            print('===== Decision Time Gap: {} ====\n '.format(self.people))
            if self.people == 'senior editors' or self.people == 'reviewing editors': 
                people_ID_sorted_keys = self.filter_people_ID()
                for bin_key in people_ID_sorted_keys: 
                    acc_time_gap_list = [ tup[2] for tup in self.bins_dict[bin_key] if tup[1]==1 ]
                    rej_time_gap_list = [ tup[2] for tup in self.bins_dict[bin_key] if tup[1]==0 ]
                    all_list = acc_time_gap_list + rej_time_gap_list

                    encourageRate = np.mean([ tup[1] for tup in self.bins_dict[bin_key] ]) 

                    print(
                        self.people, 
                        bin_key, # people ID 
                        np.mean(all_list), # Avg. Decision Gap
                        len(all_list), # Total Len
                        np.mean(acc_time_gap_list) - np.mean(all_list), # Acc Gap Delta
                        np.mean(rej_time_gap_list) - np.mean(all_list), # Rej Gap Delta
                        encourageRate, # encourageRate, for correlation analysis
                    )

            ##################################
            # Analyze Reivewers First and Return 
            ##################################
            if self.people == 'reviewers':

                ##################################
                # Plot the review histogram first 
                ##################################
                print('Total Reivewer Count', len(self.bins_dict.keys()) )
                review_num_count = []
                for people_ID in sorted(self.bins_dict.keys()):
                    review_num_count.append( len(self.bins_dict[people_ID]) ) 
                bins = list(range(0,22))
                plt.figure(figsize=(8,6))
                plt.hist(
                    review_num_count, 
                    bins=bins, 
                    label="Cumulative Distribution", 
                    #density=True, 
                    linewidth=1.5,
                    cumulative=True,
                    histtype='step') 
                plt.hist(
                    review_num_count, 
                    bins=bins, 
                    label="Reversed Cumulative Distribution", 
                    #density=True, 
                    linewidth=1.5,
                    cumulative=-1,
                    histtype='step')                     
                plt.grid()
                plt.xticks( range(0,25,5) )
                plt.xlabel("Number of Full Submissions Reviewed", size=14)
                plt.ylabel("Cumulative Count of occurrence", size=14)
                plt.title('Cumulative Histogram of Number of Full Submissions Reviewed')
                plt.legend(loc='upper right')
                plt.savefig( str(ROOT_DIR) + "review_num_count.png")


                ##################################
                # More experienced, BERT score 
                ##################################
                experience_num_data_list = [ [] for _ in bins ]
                for people_ID in sorted(self.bins_dict.keys()):
                    experience_num = len(self.bins_dict[people_ID])
                    experience_num = min( max(bins), experience_num) # for people who review too many submissions
                    experience_num_data_list[experience_num].extend(self.bins_dict[people_ID]) # merge the list - we do not have individual level analysis any more; 

                print('Experience (Num of Reviews)')
                for bin_key in bins: # number of submissions reviewed 
                    data_list = experience_num_data_list[bin_key] 
                    data_list = [ tup[1] for tup in data_list ]
                    total_len = len(data_list)
                    if total_len == 0:
                        encourageRate = 0. 
                        encourageRate_interval = 0. 
                    else:
                        encourageRate = np.mean(data_list) 
                        encourageRate_interval = 1.96 * np.std(data_list) / math.sqrt(total_len)
                    print(
                        "[ {} ]".format(bin_key), "Total [ {} ]".format(total_len) , 
                        "Encourage Rate [ {:.3f} ± {:.3f} ]".format(encourageRate, encourageRate_interval), 
                    )

                aggregate_experience_scheme = [
                    [1, 2, 3, 4, 5,],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + [21],
                ] 

                for aggregate_experience_row in aggregate_experience_scheme:
                    # print('Aggregate:', aggregate_experience_row)
                    aggregate_temp_list = []
                    for aggregate_col in aggregate_experience_row:
                        aggregate_temp_list.extend(experience_num_data_list[aggregate_col])

                    data_list = aggregate_temp_list
                    data_list = [ tup[1] for tup in data_list ]
                    total_len = len(data_list)
                    if total_len == 0:
                        encourageRate = 0. 
                        encourageRate_interval = 0. 
                    else:
                        encourageRate = np.mean(data_list) 
                        encourageRate_interval = 1.96 * np.std(data_list) / math.sqrt(total_len)
                    print(
                        "Aggregate: {}".format(aggregate_experience_row), "Total [ {} ]".format(total_len) , 
                        "Encourage Rate [ {:.3f} ± {:.3f} ]".format(encourageRate, encourageRate_interval), 
                    )
                    

                ##################################
                # The Journey of reviewing eLife
                ##################################
                print('===== The Journey of Becoming a More Experienced eLife Reviewer ====')
                journey_bins = list(range(0,3))
                journey_bin_collection = [ [] for _ in journey_bins]
                for people_ID in sorted(self.bins_dict.keys()):
                    experience_num = len(self.bins_dict[people_ID])
                    if experience_num >= len(journey_bins):
                        for journey_len in journey_bins: 
                            one_score = self.bins_dict[people_ID][journey_len][1]
                            journey_bin_collection[journey_len].append(one_score)
                
                for journey_len in journey_bins: # number of submissions reviewed 
                    data_list = journey_bin_collection[journey_len]
                    total_len = len(data_list)
                    if total_len == 0:
                        encourageRate = 0. 
                        encourageRate_interval = 0. 
                    else:
                        encourageRate = np.mean(data_list) 
                        encourageRate_interval = 1.96 * np.std(data_list) / math.sqrt(total_len)
                    print(
                        "Journey: {}".format(journey_len + 1), "Total [ {} ]".format(total_len) , 
                        "Encourage Rate [ {:.3f} ± {:.3f} ]".format(encourageRate, encourageRate_interval), 
                    )

                ##################################
                # Ranking Analysis
                # This is an independent code, not utilizing the existing methods or variables. 
                # Need to Construct a Submission Dict 
                ##################################
                print('===== Ranking Analysis of Reviews ====')

                ##################################
                # Prepare for biased simulation to match curve 
                # randomly select x% of the people, as the biased one 
                ##################################
                SIMULATION_FLAG = True
                if SIMULATION_FLAG:
                    people_ID_list = sorted(self.bins_dict.keys())
                    # permute, and extract x front, x end. 
                    permuted_people_ID_list = np.random.permutation(people_ID_list)
                    BIASED_PEOPLE_NUM = int( len(permuted_people_ID_list) * (0.05 / 2) ) # default: 0.05 ~ 5% biased 
                    bias_low_people = set(permuted_people_ID_list[:BIASED_PEOPLE_NUM])
                    bias_high_people = set(permuted_people_ID_list[-BIASED_PEOPLE_NUM:])
                    assert bias_low_people.isdisjoint(bias_high_people)


                ##################################
                # First Pass: 
                # load and parse raw data into manuscript_ID_to_reviews
                ##################################
                temp_csv = load_nlp_review_data(DATA_ROOT='./nlp/')
                manuscript_ID_to_reviews = dict()
                for i, line in tqdm(temp_csv.iterrows(), total=temp_csv.shape[0]):
                    # print('line', line )
                    if math.isnan(line['ms']):
                        continue # skip this row, problematic 
                    manuscript_ID = int(line['ms']) # Manuscript no. 
                    if manuscript_ID not in manuscript_ID_to_reviews:
                        manuscript_ID_to_reviews[manuscript_ID] = [] # init empty list 
                    
                    existing_reviewer_IDs = [ tup[1] for tup in manuscript_ID_to_reviews[manuscript_ID]]
                    if line['Reviewer ID'] not in existing_reviewer_IDs: 
                        review_BERT_score =  line['score']

                        ##################################
                        # Random Simulation: 
                        # overwrite with random scores
                        ##################################   
                        if SIMULATION_FLAG:
                            review_BERT_score = np.random.normal(loc=0.0, scale=1.0)

                            ##################################
                            # Biased Simulation
                            # reparameterization trick
                            ##################################  
                            # BIAS_OFFSET = 1.0  # default: 1.0 sigma offset 
                            BIAS_OFFSET = 0.0
                            if line['Reviewer ID'] in bias_low_people:
                                review_BERT_score -= BIAS_OFFSET
                            elif line['Reviewer ID'] in bias_high_people:
                                review_BERT_score += BIAS_OFFSET

                        manuscript_ID_to_reviews[manuscript_ID].append( [ review_BERT_score, int(line['Reviewer ID']), ] ) 
                        # there is duplicates in the file 


                ##################################
                # Second Pass: 
                # use manuscript_ID_to_reviews to build reviewer entry, with sorted index
                ##################################                
                reviewer_ID_dict = dict()
                for manuscript_ID, review_list in manuscript_ID_to_reviews.items():
                    review_list.sort(key=lambda x: x[0]) # will sort only based on scores 
                    if len(review_list) <=1: # only consider when there are at least two reviews 
                        continue 
                    score_list = [tup[0] for tup in review_list]
                    avg_score = np.mean(score_list)
                    for rank, review_entry in enumerate(review_list):
                        # unpack the entry
                        review_BERT_score, reviewer_ID = review_entry
                        # need to insert rank 
                        if reviewer_ID not in reviewer_ID_dict:
                            reviewer_ID_dict[reviewer_ID] = []
                        delta_score = review_BERT_score - avg_score # diff between avg 

                        ##################################
                        # We use normalized rank (i.e., rank divided by the number of reviews), 
                        # since papers might have different number of reviews. 
                        ##################################
                        reviewer_ID_dict[reviewer_ID].append( [rank / (len(review_list)-1), review_BERT_score, avg_score, delta_score] )

                ##################################
                # Third Pass: 
                # Case by case analysis - anlyze each reviewer with a minimum submission threshold
                ##################################
                print('reviewer heterogeneity analysis info')
                avg_rank_list = []
                for reviewer_ID, review_history_list in reviewer_ID_dict.items():

                    ##################################
                    # Filter the list 
                    # Only keep people with more than x submissions 
                    ##################################                
                    if len(review_history_list) < 5: #5: #10:
                        continue 

                    rank_list = [ tup[0] for tup in review_history_list ] # Note: rank starts from zero 
                    score_list = [ tup[1] for tup in review_history_list ]
                    avg_score_list = [ tup[2] for tup in review_history_list ]
                    delta_score_list = [ tup[3] for tup in review_history_list ]

                    num_reviews = len(rank_list)

                    ##################################
                    # Note: This will generate many lines. 
                    # Comment this line out to see other things more clearly. 
                    ##################################                

                    if num_reviews>=5: # threshold 
                        avg_rank_list.append(np.mean(rank_list))


                ##################################
                # Output simulation in a quick way
                ##################################   
                avg_rank_list.sort(reverse=True)             
                with open("output/reviewer_simulation.txt", "w") as f:
                    for sorted_value in avg_rank_list:
                        f.write(str(sorted_value) + '\n' )
                return # no more anchoring analysis 


            if self.people == 'reviewing editors':
                ##################################
                # Heterogeneity and Simulation Analysis
                ##################################

                ##################################
                # Ranking Analysis
                # This is an independent code, not utilizing the existing methods or variables. 
                # Need to Construct a Submission Dict 
                ##################################
                print('===== Reviewing Editors Heterogeneity ====')

                ##################################
                # Prepare for biased simulation to match curve 
                # randomly select x% of the people, as the biased one 
                ##################################
                SIMULATION_FLAG = True
                # SIMULATION_FLAG = False
                if SIMULATION_FLAG:
                    people_ID_list = sorted(self.bins_dict.keys())
                    # permute, and extract x front, x end. 
                    permuted_people_ID_list = np.random.permutation(people_ID_list)

                    BIASED_PEOPLE_PERCENT = 0.1 # Best Matched value 0.1

                    BIASED_PEOPLE_NUM = int( len(permuted_people_ID_list) * ( BIASED_PEOPLE_PERCENT / 2) )
                    bias_low_people = set(permuted_people_ID_list[:BIASED_PEOPLE_NUM])
                    bias_high_people = set(permuted_people_ID_list[-BIASED_PEOPLE_NUM:])
                    assert bias_low_people.isdisjoint(bias_high_people)
                

                ##################################
                # Output simulation in a quick way
                ##################################
                avg_acc_rate_list = []
                tmp_all_decisions = []
                for people_ID in self.bins_dict.keys():

                    ##################################
                    # Length filtering
                    ##################################
                    total_len = len(self.bins_dict[people_ID])
                    if total_len < 20:
                        continue 

                    ##################################
                    # Simulate each outcome
                    # or, get the real value 
                    ##################################
                    bit_array = [ t[1] for t in self.bins_dict[people_ID]] # real
                    acc_rate = np.mean(bit_array) # real  # tmp_all_decisions mean std 0.5626210458360232 0.49606310547994065

                    if SIMULATION_FLAG:
                        reviewing_editor_avg_acc = 0.5626210458360232
                        p_acc = reviewing_editor_avg_acc
                        
                        BIAS_REVIEWING_OFFSET = 0.20 # Default value 0.20

                        if people_ID in bias_low_people:
                            p_acc -= BIAS_REVIEWING_OFFSET
                        elif people_ID in bias_high_people:
                            p_acc += BIAS_REVIEWING_OFFSET

                        simulated_acc_count = np.random.binomial( len(bit_array), p_acc, 1)[0]
                        acc_rate = simulated_acc_count / len(bit_array) # simulated 
                        

                    ##################################
                    # And then calculate the overall stats
                    # need to grab the total stats here! 
                    ##################################
                    avg_acc_rate_list.append( acc_rate )
                    tmp_all_decisions.extend(bit_array) 
                
                ##################################
                # Output simulation in a quick way
                ##################################   
                print('tmp_all_decisions mean std', np.mean(tmp_all_decisions), np.std(tmp_all_decisions))
                avg_acc_rate_list.sort(reverse=True)             
                with open("output/reviewing_editor_simulation.txt", "w") as f:
                    for sorted_value in avg_acc_rate_list:
                        f.write(str(sorted_value) + '\n' )

            ##################################
            # Anchoring Effect Starts
            ##################################
            print('===== Anchoring Effect: {} ====\n '.format(self.people))

            people_ID_sorted_keys = self.filter_people_ID()


            rej_cnt_list = [ AnchoringEffectCounter(prefix_zeros_len=prefix_zeros_len, choice='reject') for prefix_zeros_len in [1,2,3,4] ]
            acc_cnt_list = [ AnchoringEffectCounter(prefix_zeros_len=prefix_zeros_len, choice='accept') for prefix_zeros_len in [1,2,] ]
            multi_test_p_value_list = []
            for bin_key in people_ID_sorted_keys: 
                ##################################
                # Sort on initial decision date. Break tie using manuscript ID. 
                ##################################

                ########################
                # year filtering 
                ########################
                bit_array = []
                for t in self.bins_dict[bin_key]:
                    date_obj=t[0]
                    bit_array.append(t[1])
                
                total_len = len(bit_array)

                if self.people == 'reviewing editors' or self.people == 'senior editors':
                    if total_len<100:
                        continue 
                    # pass 
                elif self.people == 'reviewers':
                    if total_len<20:
                        continue 

                
                encourageRate = sum(bit_array) / total_len


                print(
                    self.people, 
                    bin_key, # people ID 
                    total_len, # total number of submissions
                    encourageRate, # base encourage rate - all 
                    rej_cnt_list[0].update(bit_array), 
                    rej_cnt_list[1].update(bit_array), 
                    rej_cnt_list[2].update(bit_array), 
                    rej_cnt_list[3].update(bit_array), 
                    acc_cnt_list[0].update(bit_array), 
                    acc_cnt_list[1].update(bit_array), 
                )


            ##################################
            # Anchoring Effect Summary: Report Aggregate-Level Results 
            ##################################
            print(
                '===== Anchoring Effect Summary: {} ====\n '.format(self.people), 
                "After_{}_{} [ {:.3f} ± {:.3f} ]\n ".format( 1, 'reject', * rej_cnt_list[0].summary() ), 
                "After_{}_{} [ {:.3f} ± {:.3f} ]\n ".format( 2, 'reject', * rej_cnt_list[1].summary() ),
                "After_{}_{} [ {:.3f} ± {:.3f} ]\n ".format( 3, 'reject', * rej_cnt_list[2].summary() ), 
                "After_{}_{} [ {:.3f} ± {:.3f} ]\n ".format( 4, 'reject', * rej_cnt_list[3].summary() ), 

                "After_{}_{} [ {:.3f} ± {:.3f} ]\n ".format( 1, 'accept', * acc_cnt_list[0].summary() ), 
                "After_{}_{} [ {:.3f} ± {:.3f} ]\n ".format( 2, 'accept', * acc_cnt_list[1].summary() ), 
            )

            ##################################
            # Anchoring Effect Ends 
            ##################################



            ##################################
            # Weekend Effect Individual Analysis 
            ##################################
            print('===== Weekend Effect Individual Analysis: {} ===='.format(self.people))
            multi_test_p_value_list = []
            for people_ID in people_ID_sorted_keys: 
                ########################
                # year filtering 
                ########################
                weekday_bitarray = []
                weekend_bitarray = []
                for t in self.bins_dict[people_ID]:
                    date_time_obj = t[0]
                    weekday = date_time_obj.weekday() # 0-6 
                    if weekday == 5 or weekday == 6 : # Saturday, Sunday
                        # 'weekends'
                        weekend_bitarray.append(t[1])
                    else:
                        # 'weekdays'
                        weekday_bitarray.append(t[1])

                weekday_len = len(weekday_bitarray)
                weekend_len = len(weekend_bitarray)
                total_len = weekday_len + weekend_len
                if total_len < 100:
                    continue 

                weekday_encourageRate = np.mean(weekday_bitarray)
                weekend_encourageRate = np.mean(weekend_bitarray)
                total_bitarray = weekday_bitarray + weekend_bitarray

                ##################################
                # Chi-square test of independence of variables in a contingency table.
                ##################################

                # Construct the contingency table first 
                obs = np.array([
                    [ np.sum(weekday_bitarray)  , weekday_len - np.sum(weekday_bitarray) ], 
                    [ np.sum(weekend_bitarray)  , weekend_len - np.sum(weekend_bitarray) ], 
                    ]
                    )

                if np.all(obs>=5):
                    # 2-way classification: Contingency test
                    chi2, p_value, dof, expctd = chi2_contingency(obs)
                    assert dof == 1
                    multi_test_p_value_list.append(p_value)
                    if p_value < 0.1: # following James' suggestion: 0.001
                        # print('2-way classification: Contingency test')
                        print( 'ID:', people_ID, 'chi2 {:.3f}, p_value {:.3f} Total {}'.format(chi2, p_value, total_len) )
                        # print('encourageRate {:.3f}'.format(encourageRate), 
                        # 'Condition on Rej {:.3f}'.format(rej_cnt_list[0].update(bit_array)), 
                        # 'Condition on Acc {:.3f}'.format(acc_cnt_list[0].update(bit_array)), 
                        # )
                        # pass 
                        print('obs', obs)
                        print(
                            " {} ".format(people_ID), 
                            "Weekday_Total {:d} {:.3f}".format(weekday_len, weekday_len/total_len) , 
                            "Weekend_Total {:d} {:.3f}".format(weekend_len, weekend_len/total_len) , 
                            "Weekday_Encourage_Rate {:.3f} ".format(weekday_encourageRate), 
                            "Weekend_Encourage_Rate {:.3f} ".format(weekend_encourageRate), 
                            "Avg_Encourage_Rate {:.3f} ".format( np.mean(total_bitarray) ), 
                        )

                    else:
                        # print('No Significant Anchoring Effect: p_value {:.3f}'.format(p_value) )
                        pass
                 
                else:
                    # print('Warning: Insufficient values in bins', obs)
                    pass

            ##################################
            # Collect an array of p-values, and perform 
            # FDR: false decovery rate control. 
            ##################################
            # note: reviewing editor does not have sufficient data 
            if self.people == 'senior editor':
                # note: reviewing editor does not have sufficient data 
                import statsmodels.stats.multitest as smt
                multi_test_ret = smt.multipletests(multi_test_p_value_list, alpha=0.1, method='hs', is_sorted=False, returnsorted=False)
                print('multi_test_p_value_list', multi_test_p_value_list)
                print('multi_test_ret', multi_test_ret)

            return 

        ##################################
        # Step 2: Summary
        ##################################

        for bin_key in sorted(self.bins_dict.keys()):
            total_len = len(self.bins_dict[bin_key])
            # self.bins_dict[bin_key].sort() # sort based on date 

            data_list = [ tup[1] for tup in self.bins_dict[bin_key] ]
            total_len = len(data_list)
            if total_len == 0:
                encourageRate = 0. 
                encourageRate_interval = 0. 
            else:
                encourageRate = np.mean(data_list) 
                encourageRate_interval = 1.96 * np.std(data_list) / math.sqrt(total_len)
            # encourageRate = sum([ tup[1] for tup in self.bins_dict[bin_key] ]) / total_len
            # encourageRate_interval = 1.96 * math.sqrt( (encourageRate * (1 - encourageRate)) / total_len)
            print(
                # self.people, 
                self.aggregate_field, " [ {} ]".format(bin_key), "Total [ {:4d} ]".format(total_len) , 
                "Encourage Rate [ {:.3f} ± {:.3f} ]".format(encourageRate, encourageRate_interval), 
            )
        

        ##################################
        # Additional Analysis: Decision Time Gap Study 
        ##################################
        if self.aggregate_field == 'all':
            # only do it once 
            acc_time_gap_list = [ tup[2] for tup in self.bins_dict[bin_key] if tup[1]==1 ]
            rej_time_gap_list = [ tup[2] for tup in self.bins_dict[bin_key] if tup[1]==0 ]

            print('Acc Time Gap: mean {:.3f}, std {:.3f}'.format(np.mean(acc_time_gap_list), np.std(acc_time_gap_list)) )
            print('Rej Time Gap: mean {:.3f}, std {:.3f}'.format(np.mean(rej_time_gap_list), np.std(rej_time_gap_list)) )

            # bins = np.linspace(0,14)
            bins = np.linspace(0,60, num=12)
            plt.figure(figsize=(8,6))
            plt.hist(
                acc_time_gap_list, 
                bins=bins, 
                alpha=0.5, 
                # label="acceptance", density=True, 
                label="encourage", density=True, 
                histtype='bar') 
            plt.hist(
                rej_time_gap_list, 
                bins=bins, 
                alpha=0.5, 
                label="reject", density=True, 
                histtype='bar') 
            plt.xlabel("time gap (day) from submission received to decided", size=24)
            plt.ylabel("Probability Density", size=24)
            plt.legend(loc='upper right', prop={'size': 24})

            plt.savefig( str(ROOT_DIR) + "global_time_gap.png")


        ##################################
        # Plot Busyness effect with error bars 
        ##################################
        if self.aggregate_field == 'year_month':
            month_to_rates_list = defaultdict(list)
            for bin_key in sorted(self.bins_dict.keys()):
                month, year = bin_key.split()
                month, year = int(month), int(year) 
                total_len = len(self.bins_dict[bin_key])

                data_list = [ tup[1] for tup in self.bins_dict[bin_key] ]
                total_len = len(data_list)
                if total_len == 0:
                    encourageRate = 0. 
                    encourageRate_interval = 0. 
                else:
                    encourageRate = np.mean(data_list) 
                    encourageRate_interval = 1.96 * np.std(data_list) / math.sqrt(total_len)

                month_to_rates_list[month].append((total_len, encourageRate))
            

            x = list(range(1,12+1))
            y1 = []
            y1_err = []
            y2 = []
            y2_err = []

            for month in range(1,12+1):
                len_list = [ tup[0] for tup in month_to_rates_list[month] ]
                encourageRate_list = [ tup[1] for tup in month_to_rates_list[month] ]

                y1.append(np.mean(len_list))
                y1_err.append(np.std(len_list))
                y2.append(np.mean(encourageRate_list))
                y2_err.append(np.std(encourageRate_list))

            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()
            lns1 = ax1.errorbar(x, y1, yerr=y1_err, color='red', label = 'Number of Submissions')[0]
            ax1.legend(loc="upper left")
            ax1.set_xticks(x)

            lns2 = ax2.errorbar(x, y2, yerr=y2_err, color='blue', label = 'Encouragement Rate')
            ax2.legend(loc="upper right")

            ax1.set_xlabel('Month')
            ax1.set_ylabel('Number of Submissions')
            ax2.set_ylabel('Encouragement Rate')
            ax2.set_ylim((0.0, 1.0))

            plt.title('Business Effect on {}'.format(self.people) )

            # plt.show()
            plt.savefig( str(ROOT_DIR) + "busyness.png")

        return 




class Submission_History_Iterator:

    def __init__(self, people : str, ):

        ##################################
        # load submission history
        # Load submission decision history 
        # Currently: 34,161 (exactly matching the doc) 
        ##################################
        self.people = people

        if people == 'reviewers':
            self.main_csv = load_nlp_review_data(DATA_ROOT='./nlp/')
            pass
        else:
            self.main_csv = pd.read_csv("./eLife_Paper_history_2019_03_15.csv",
                                error_bad_lines=False)

        return 


    def get_accept_score(self, line, x_people):

        ret_flag = None 

        if x_people.people == 'reviewing editors': 
        # if True:
            ##################################
            # Full paper Decisions 
            ##################################
            full_decision_str_to_id = {
                'Reject Full Submission': 0.,
                # 'Revise Full Submission': 0., #
                'Revise Full Submission': 1., # 
                'Accept Full Submission': 1.,
            }
        
            assert line['full_decision'] in full_decision_str_to_id, line['full_decision']
            for dc_field in ['full_decision', 'rev1_decision', 'rev2_decision', 'rev3_decision', 'rev4_decision']: # consider revisions
                dc = line[dc_field]
                if isinstance(dc, str) and dc in full_decision_str_to_id:
                    ret_flag = full_decision_str_to_id[line[dc_field]]

        elif x_people.people == 'senior editors': 
            ##################################
            # Definition of acceptance 
            # Init Decisions 
            ##################################
            decision_str_to_id = {
                'Reject Initial Submission': 0.,
                'Encourage Full Submission': 1.,
                # 'Reject Full Submission': 0.,
            }
            assert line['initial_decision'] in decision_str_to_id
            # senior_editor_dict[senior_editor_id].append((date_time_obj, decision_str_to_id[line['initial_decision']]))
            ret_flag = decision_str_to_id[line['initial_decision']]

        elif x_people.people == 'reviewers': 
            ##################################
            # Use normalized continuous BERT score 
            ##################################
            ret_flag = float(line['score']) / 5.0 # BERT score 
            assert ret_flag >= 0. and ret_flag <= 1.0 , ret_flag

        else:
            raise NotImplementedError
        return ret_flag

    
    def iterate(self, aggregate_field_list : list, **kwargs):
        
        people = self.people

        x_people = People_Meta_Data(people=people)

        print("Working on people:", people)

        ##################################
        # Allocate aggregators 
        ##################################
        
        list_EntryAggregator = [
            EntryAggregator(aggregate_field, x_people.people) for aggregate_field in aggregate_field_list
        ]
        

        for i, line in tqdm(self.main_csv.iterrows(), total=self.main_csv.shape[0]):
            
            ##################################
            # Skip problematic rows 
            # simple withdraw 
            # one errorneous row 
            ##################################
            if x_people.people == 'reviewing editors' or x_people.people == 'senior editors': 
                if math.isnan(line['senior_editor']) or not isinstance(line['initial_decision'], str) or not isinstance(line['initial_decision_dt'], str):
                    continue
                if line['initial_decision'] == 'Simple Withdraw':
                    continue 
                if line['initial_decision'] == 'Reject Full Submission':
                    # print(line)
                    continue 


            ##################################
            # Filtering Condition
            ##################################

            if x_people.people == 'reviewing editors': 

                ##################################
                # Collect data stratified by reviewing editors 
                ##################################
                if not math.isnan(line['reviewing_editor']) and \
                    isinstance(line['full_decision'], str) and \
                        isinstance(line['full_decision_dt'], str) and\
                            line['full_decision']!='Simple Withdraw': # conservative? 
                    
                    date_time_str = line['full_decision_dt']

                    ##################################
                    # Overwrite, only to see the quality check difference 
                    ##################################

                    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
                    people_ID = int(line['reviewing_editor'])

                    date_time_str = line['full_qc_dt']
                    qc_date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')

                    if qc_date_time_obj <= date_time_obj:
                        
                        if (date_time_obj-qc_date_time_obj).days > 100:
                            pass

                        
                    else:
                        continue



                else:
                    continue # submission does not get encouraged 

            elif x_people.people == 'senior editors': 

                ##################################
                # Pre-process each data field
                ##################################
                date_time_str = line['initial_decision_dt']

                ##################################
                # Overwrite, only to see the quality check difference 
                ##################################
                
                date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
                people_ID = int(line['senior_editor'])

                date_time_str = line['initial_qc_dt']
                qc_date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
                assert qc_date_time_obj <= date_time_obj

            elif x_people.people == 'reviewers': 

                ##################################
                # Pre-process each data field
                ##################################
                if not isinstance(line['full_qc_dt'], str):
                    print("Warning: No full_qc_dt", line)
                    continue # skip this line
                date_time_str = line['full_qc_dt'] # full_qc_dt
                date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
                people_ID = int(line['Reviewer ID'])

                # unfortunately we do not have the time stamp to tell when each review is submitted. 
                qc_date_time_obj = date_time_obj
            else: 
                raise NotImplementedError

            ##################################
            # Global Time Filtering Condition: only consider 2016++, i.e., discarding 2015 & earlier 
            ##################################

            if date_time_obj.year<=2015: # only do 2016, 2017, 2018 
                continue 
            
            ##################################
            # General Supporting Fields  
            # if not found, should continue 
            ##################################

            people_query_ret = x_people.query_all(people_ID=people_ID)
            if people_query_ret is None: # not found
                print('Warning: people_ID', people_ID, 'Not Found')
                continue  # have to skip this entry 

            accept_score = self.get_accept_score(line, x_people)
            if accept_score is None: # not found
                print('Warning: people_ID', people_ID, 'accept_score Not Found')
                continue # no such field 



            ##################################
            # Collect data stratified by Y axis 
            # many options; do all
            ##################################

            
            for entry_agg in list_EntryAggregator:
                entry_agg.update(
                    people_query_ret = people_query_ret,
                    accept_score = accept_score,
                    date_time_obj = date_time_obj,
                    qc_date_time_obj = qc_date_time_obj, 
                    )



        ##################################
        # Summary Collected data 
        ##################################
        for entry_agg in list_EntryAggregator:
            entry_agg.summary()

    
        return 



def main():

    aggregate_field_list = [
        'weekday', 'weekend', 
        'year_month', 'month', 'season',  'year', 
        'country_code', 'race', 'race & gender', 
        'gender', 'US', 'people_ID', 'all', 
        ] # 


    submission_iter = Submission_History_Iterator(
        # people='senior editors',
        # people='reviewing editors',
        people='reviewers',
    )
    submission_iter.iterate(
        aggregate_field_list = aggregate_field_list, 
        )

    return 

if __name__ == "__main__":
    main()
    pass 