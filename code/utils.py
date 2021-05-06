import pandas as pd
import os
from collections import Counter
import re
import json # for parsing fields
import math
import matplotlib.pyplot as plt
import time
import csv
import pickle
from tqdm import tqdm
import numpy as np

import datetime




from fuzzywuzzy import fuzz
from fuzzywuzzy import process

class Insitution_Fuzzy_Mather():


    def __init__(self): 
        ##################################
        # Prepare insitution json file  
        ##################################
        self.name_choices = [ ]
        self.domains_choices = [ ]
        self.map_domain_to_name = dict()
        self.map_name_to_dict = dict()

        
        with open("./lib/world_universities_and_domains.json", encoding='utf-8') as json_file:
            univerisity_json = json.load(json_file)
            # Example: [
            # 	...
            # 	{
            # 	    "alpha_two_code": "TR",
            # 	    "country": "Turkey",
            # 	    "state-province": null,
            # 	    "domains": [
            # 	        "sabanciuniv.edu",
            # 	        "sabanciuniv.edu.tr"
            # 	    ],
            # 	    "name": "Sabanci University",
            # 	    "web_pages": [
            # 	        "http://www.sabanciuniv.edu/",
            # 	        "http://www.sabanciuniv.edu.tr/"
            # 	    ],
            # 	},
            # 	...
            # ]

        for univ_dict in univerisity_json:
            name = univ_dict['name'] 
            domains = univ_dict['domains'] 

            ##################################
            # Extract choices  
            ##################################
            self.name_choices.append(name)
            self.domains_choices.extend(domains)
            for d in domains:
                self.map_domain_to_name[d] = name 
            self.map_name_to_dict[name] = univ_dict

        return  


    def match(self, email, institution_name):
        ##################################
        # Match two fileds at the same time 
        # The match returns a list of tuples 
        ##################################


        ##################################
        # Match institution name 
        ##################################
        match_name = '' 
        alpha_two_code = ''
        if institution_name in self.name_choices:
            match_name = institution_name

        if isinstance(email, str) and 'gmail.com' not in email and 'mac.com' not in email: 
            # elifesciences.org
            email_domain = email.split('@')[-1] # 
            email_match_ret = process.extractOne(email_domain, self.domains_choices) 
            email_db, email_confidence = email_match_ret

            if email_confidence >=93:
                match_name = self.map_domain_to_name[email_db]
            


        ##################################
        # Only do 100% correct translation 
        ##################################
        if match_name != '':
            alpha_two_code = self.map_name_to_dict[match_name]['alpha_two_code']
            
                
        return match_name, alpha_two_code


# from ethnicolr import census_ln
def guess_race(full_name):
    if isinstance(full_name, str): 
        # only handle this case 
        names = [{'name': full_name.split()[-1]}]
        tmp_df = pd.DataFrame(names)
        tmp_out = census_ln(tmp_df, 'name')

        this_race = 'Fail' 
        race_confidence = 0.0
        output_list = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']
        for race_str in output_list:
            percentage = tmp_out[race_str][0]
            if isinstance(percentage, str) and percentage != '(S)':
                if float(percentage) > race_confidence:
                    this_race = race_str[3:] # exclude pct 
                    race_confidence = float(percentage) # pick max confident one 
    else:
        this_race = 'error'
    return this_race



def get_senior_meta_data():
    """
    docstring
    """

    import gender_guesser.detector as gender
    d = gender.Detector()
    genders = []

    ##################################
    # Load senior editor meta-data
    ##################################

    senior_csv = pd.read_csv("./eLife_Senior_editors_Distinct.csv", sep=';', 
                        error_bad_lines=False)

    formatted_senior_info_dict = dict()

    fuzzy_matcher = Insitution_Fuzzy_Mather()

    alpha_two_code_list_DEBUG = []
    races = []



    for i, line in tqdm(senior_csv.iterrows(), total=senior_csv.shape[0]):
        
        assert line['Senior editor ID'] not in formatted_senior_info_dict

        first_name = line['Senior editor name'].split()[0] 

        institution_name = line['Senior editor institution']
        email = line['Senior editor email']

        match_name, alpha_two_code = fuzzy_matcher.match(email=email, institution_name=institution_name)

        this_race = guess_race(line['Senior editor name'])
        this_gender = d.get_gender( first_name )


        formatted_senior_info_dict[line['Senior editor ID']] = {
            'people_ID': line['Senior editor ID'], 
            'name': line['Senior editor name'],
            'institution': line['Senior editor institution'], 
            'email': line['Senior editor email'], 
            'race': this_race, 
            'country_code': alpha_two_code, # -2 
            'gender': this_gender, # -1 
        }
        

        alpha_two_code_list_DEBUG.append(alpha_two_code)
        races.append(this_race)
        genders.append(this_gender)

    ##################################
    # Print Race prediction stats
    ##################################

    print("race error count", races.count('error'), "len", len(races))
    print("race Fail count", races.count('Fail'))
    total_race_success = len(races) - races.count("error") - races.count("Fail")
    print("total_race_success", total_race_success)
    print("white", races.count("white"), races.count("white")/total_race_success)
    print("black", races.count("black"), races.count("black")/total_race_success)
    print("api", races.count("api"), races.count("api")/total_race_success)
    print("aian", races.count("aian"), races.count("aian")/total_race_success)
    print("2prace", races.count("2prace"), races.count("2prace")/total_race_success)
    print("hispanic", races.count("hispanic"), races.count("hispanic")/total_race_success)


    ##################################
    # Print Country aggregation stats
    ##################################

    for key in set(alpha_two_code_list_DEBUG):
        occurance = alpha_two_code_list_DEBUG.count(key)
        print(key, '\t', occurance)

    ##################################
    # Print Gender prediction stats
    ##################################
    print("gender male:", genders.count('male'), "female:", genders.count('female'), "Total:", len(genders))
    return formatted_senior_info_dict


def get_reviewing_editor_meta_data():
    """
    docstring
    """

    import gender_guesser.detector as gender
    d = gender.Detector()
    genders = []

    ##################################
    # Load reviewing editor meta-data
    ##################################

    senior_csv = pd.read_csv("./eLife_Reviewing_editors_Distinct.csv", sep='	', 
                        error_bad_lines=False)

    formatted_reviewing_editor_info_dict = dict()

    alpha_two_code_list_DEBUG = []
    fuzzy_matcher = Insitution_Fuzzy_Mather()

    races = []


    for i, line in tqdm(senior_csv.iterrows(), total=senior_csv.shape[0]):
        
        assert line['Reviewing editor ID'] not in formatted_reviewing_editor_info_dict

        
        if not isinstance(line['Reviewing editor name'], str):
            print(line) 
            line['Reviewing editor name'] = 'AAA BBB'

        first_name = line['Reviewing editor name'].split()[0] 


        email = line['Reviewing editor email']
        institution_name = line['Reviewing editor institution']
        match_name, alpha_two_code = fuzzy_matcher.match(email=email, institution_name=institution_name)

        this_race = guess_race(line['Reviewing editor name'])
        this_gender = d.get_gender( first_name )


        formatted_reviewing_editor_info_dict[line['Reviewing editor ID']] = {
            'people_ID': line['Reviewing editor ID'],
            'name': line['Reviewing editor name'],
            'institution': line['Reviewing editor institution'], 
            'email': line['Reviewing editor email'], 
            'race': this_race, 
            'country_code': alpha_two_code, # -2 
            'gender': this_gender, # -1 
        }

        alpha_two_code_list_DEBUG.append(alpha_two_code)
        races.append(this_race)
        genders.append(this_gender)

    ##################################
    # Print Race prediction stats
    ##################################

    print("race error count", races.count('error'), "len", len(races))
    print("race Fail count", races.count('Fail'))
    total_race_success = len(races) - races.count("error") - races.count("Fail")
    print("total_race_success", total_race_success)
    print("white", races.count("white"), races.count("white")/total_race_success)
    print("black", races.count("black"), races.count("black")/total_race_success)
    print("api", races.count("api"), races.count("api")/total_race_success)
    print("aian", races.count("aian"), races.count("aian")/total_race_success)
    print("2prace", races.count("2prace"), races.count("2prace")/total_race_success)
    print("hispanic", races.count("hispanic"), races.count("hispanic")/total_race_success)


    ##################################
    # Print Country aggregation stats
    ##################################
    print('failure count', alpha_two_code_list_DEBUG.count(''), 'total', len(alpha_two_code_list_DEBUG))

    for key in set(alpha_two_code_list_DEBUG):
        occurance = alpha_two_code_list_DEBUG.count(key)
        print(key, '\t', occurance)
        pass 

    ##################################
    # Print Gender prediction stats
    ##################################
    print("gender male:", genders.count('male'), "female:", genders.count('female'), "Total:", len(genders))

    return formatted_reviewing_editor_info_dict

