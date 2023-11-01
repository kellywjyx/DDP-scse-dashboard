import pickle
import pandas as pd
import re
import numpy as np

def parse_gs_publ(df):
    count_patents = 0
    count_journals = 0
    count_conf = 0
    count_books = 0
    count_none = 0
    list_patents = []
    list_journals = []
    list_conf = []
    list_book = []
    list_none = []
    dict_parsed = {}
    names = sorted(df['Name'].tolist())
    
    for num in range(len(names)):
        name = names[num]
        abstracts = pickle.load(open(f"./gs_publication_set/gs_publications_{num}.pkl", "rb"))
        dict_parsed[name] = {}
        for i in range(len(abstracts)):
            dict_parsed[name][i] = {}
            try:
                title = abstracts[i].find('a',class_="gsc_oci_title_link").text
                gs_specific_url = abstracts[i].find('a',class_="gsc_oci_title_link")['href']
            except:
                try:
                    title = abstracts[i].find('div',id="gsc_oci_title").text
                    gs_specific_url = None
                except:
                    title = abstracts[i].find("meta",property="og:title",content=True).attrs['content']
                    gs_specific_url = None
            dict_parsed[name][i]['title'] = title
            dict_parsed[name][i]['gs_specific_url'] = gs_specific_url

            try:
                pdf = abstracts[i].find('div',class_="gsc_oci_title_ggi").text
                pdf_url = abstracts[i].find('div',class_="gsc_oci_title_ggi")['href']
            except:
                pdf = None
                pdf_url = None
            dict_parsed[name][i]['pdf'] = pdf
            dict_parsed[name][i]['pdf_url'] = pdf_url
            fields = abstracts[i].find_all('div', class_="gsc_oci_field")
            values = abstracts[i].find_all('div', class_="gsc_oci_value")

            list_fields = []
            for j in range(len(fields)):
                list_fields.append(fields[j].text)
            if 'Patent office' in list_fields:
                count_patents += 1
                list_patents.extend(list_fields)
                dict_parsed[name][i]['type'] = 'Patent'
                for k in range(len(fields)):
                    field_type = fields[k].text
                    if field_type=='Inventors':
                        author_names = values[k].text
                        dict_parsed[name][i]['authors'] = author_names
                    elif field_type=='Publication date':
                        date = values[k].text
                        dict_parsed[name][i]['date'] = date
                    elif field_type=='Patent office':
                        patent = values[k].text
                        dict_parsed[name][i]['patent'] = patent
                    elif field_type=='Application number':
                        application = values[k].text
                        dict_parsed[name][i]['application'] = application
                    elif field_type=='Patent number':
                        number = values[k].text
                        dict_parsed[name][i]['patent_number'] = number
                    elif field_type=='Description':
                        desc = values[k].text
                        dict_parsed[name][i]['description'] = desc
                    elif field_type=='Total citations':
                        list_years = values[k].find_all('span',class_="gsc_oci_g_t")
                        list_years.reverse()
                        num_years = len(list_years)
                        list_values = values[k].find_all('a',class_="gsc_oci_g_a")
                        list_values.reverse() # reversed, so that z-index follows order
                        list_res = []
                        l = 0
                        m = 0
                        while l < num_years:
                            ind = int(list_values[m]['style'].split('z-index:')[1])
                            if l+1 == ind:
                                list_res.append((list_years[l].text,int(list_values[m].text)))
                                m += 1
                            else:
                                list_res.append((list_years[l].text,0))
                            l += 1
                        dict_parsed[name][i]['citations_years'] = list_res
                        dict_parsed[name][i]['citations'] = int(values[k].find('a').text.strip('Cited by '))
            elif 'Journal' in list_fields:
                count_journals += 1
                list_journals.extend(list_fields)
                dict_parsed[name][i]['type'] = 'Journal'
                for k in range(len(fields)):
                    field_type = fields[k].text
                    if field_type=='Authors':
                        author_names = values[k].text
                        dict_parsed[name][i]['authors'] = author_names
                    elif field_type=='Publication date':
                        date = values[k].text
                        dict_parsed[name][i]['date'] = date
                    elif field_type=='Journal':
                        journal = values[k].text
                        dict_parsed[name][i]['journal'] = journal
                    elif field_type=='Volume':
                        volume = values[k].text
                        dict_parsed[name][i]['volume'] = volume
                    elif field_type=='Pages':
                        pages = values[k].text
                        dict_parsed[name][i]['pages'] = pages
                    elif field_type=='Issue':
                        issue = values[k].text
                        dict_parsed[name][i]['issue'] = issue
                    elif field_type=='Publisher':
                        publisher = values[k].text
                        dict_parsed[name][i]['publisher'] = publisher
                    elif field_type=='Description':
                        desc = values[k].text
                        dict_parsed[name][i]['description'] = desc
                    elif field_type=='Total citations':
                        list_years = values[k].find_all('span',class_="gsc_oci_g_t")
                        list_years.reverse()
                        num_years = len(list_years)
                        list_values = values[k].find_all('a',class_="gsc_oci_g_a")
                        list_values.reverse() # reversed, so that z-index follows order
                        list_res = []
                        l = 0
                        m = 0
                        while l < num_years:
                            ind = int(list_values[m]['style'].split('z-index:')[1])
                            if l+1 == ind:
                                list_res.append((list_years[l].text,int(list_values[m].text)))
                                m += 1
                            else:
                                list_res.append((list_years[l].text,0))
                            l += 1
                        dict_parsed[name][i]['citations_years'] = list_res
                        dict_parsed[name][i]['citations'] = int(values[k].find('a').text.strip('Cited by '))
            elif 'Conference' in list_fields:
                count_conf += 1
                list_conf.extend(list_fields)
                dict_parsed[name][i]['type'] = 'Conference'
                for k in range(len(fields)):
                    field_type = fields[k].text
                    if field_type=='Authors':
                        author_names = values[k].text
                        dict_parsed[name][i]['authors'] = author_names
                    elif field_type=='Publication date':
                        date = values[k].text
                        dict_parsed[name][i]['date'] = date
                    elif field_type=='Conference':
                        conference = values[k].text
                        dict_parsed[name][i]['conference'] = conference
                    elif field_type=='Volume':
                        volume = values[k].text
                        dict_parsed[name][i]['volume'] = volume
                    elif field_type=='Pages':
                        pages = values[k].text
                        dict_parsed[name][i]['pages'] = pages
                    elif field_type=='Publisher':
                        publisher = values[k].text
                        dict_parsed[name][i]['publisher'] = publisher
                    elif field_type=='Issue':
                        issue = values[k].text
                        dict_parsed[name][i]['issue'] = issue
                    elif field_type=='Description':
                        desc = values[k].text
                        dict_parsed[name][i]['description'] = desc
                    elif field_type=='Total citations':
                        list_years = values[k].find_all('span',class_="gsc_oci_g_t")
                        list_years.reverse()
                        num_years = len(list_years)
                        list_values = values[k].find_all('a',class_="gsc_oci_g_a")
                        list_values.reverse() # reversed, so that z-index follows order
                        list_res = []
                        l = 0
                        m = 0
                        while l < num_years:
                            ind = int(list_values[m]['style'].split('z-index:')[1])
                            if l+1 == ind:
                                list_res.append((list_years[l].text,int(list_values[m].text)))
                                m += 1
                            else:
                                list_res.append((list_years[l].text,0))
                            l += 1
                        dict_parsed[name][i]['citations_years'] = list_res
                        dict_parsed[name][i]['citations'] = int(values[k].find('a').text.strip('Cited by '))
            elif 'Book' in list_fields:
                count_books += 1
                list_book.extend(list_fields)
                dict_parsed[name][i]['type'] = 'Book'
                for k in range(len(fields)):
                    field_type = fields[k].text
                    if field_type=='Authors':
                        author_names = values[k].text
                        dict_parsed[name][i]['authors'] = author_names
                    elif field_type=='Publication date':
                        date = values[k].text
                        dict_parsed[name][i]['date'] = date
                    elif field_type=='Book':
                        book = values[k].text
                        dict_parsed[name][i]['book'] = book
                    elif field_type=='Pages':
                        pages = values[k].text
                        dict_parsed[name][i]['pages'] = pages
                    elif field_type=='Volume':
                        volume = values[k].text
                        dict_parsed[name][i]['volume'] = volume
                    elif field_type=='Publisher':
                        publisher = values[k].text
                        dict_parsed[name][i]['publisher'] = publisher
                    elif field_type=='Description':
                        desc = values[k].text
                        dict_parsed[name][i]['description'] = desc
                    elif field_type=='Total citations':
                        list_years = values[k].find_all('span',class_="gsc_oci_g_t")
                        list_years.reverse()
                        num_years = len(list_years)
                        list_values = values[k].find_all('a',class_="gsc_oci_g_a")
                        list_values.reverse() # reversed, so that z-index follows order
                        list_res = []
                        l = 0
                        m = 0
                        while l < num_years:
                            ind = int(list_values[m]['style'].split('z-index:')[1])
                            if l+1 == ind:
                                list_res.append((list_years[l].text,int(list_values[m].text)))
                                m += 1
                            else:
                                list_res.append((list_years[l].text,0))
                            l += 1
                        dict_parsed[name][i]['citations_years'] = list_res
                        dict_parsed[name][i]['citations'] = int(values[k].find('a').text.strip('Cited by '))
            else: #'Report number'
                count_none += 1
                list_none.extend(list_fields)
                dict_parsed[name][i]['type'] = 'Others'
                for k in range(len(fields)):
                    field_type = fields[k].text
                    if (field_type=='Authors') or (field_type=='Inventors'):
                        author_names = values[k].text
                        dict_parsed[name][i]['authors'] = author_names
                    elif field_type=='Publication date':
                        date = values[k].text
                        dict_parsed[name][i]['date'] = date
                    elif field_type=='Publisher':
                        publisher = values[k].text
                        dict_parsed[name][i]['publisher'] = publisher
                    elif field_type=='Volume':
                        volume = values[k].text
                        dict_parsed[name][i]['volume'] = volume
                    elif field_type=='Pages':
                        pages = values[k].text
                        dict_parsed[name][i]['pages'] = pages
                    elif field_type=='Description':
                        desc = values[k].text
                        dict_parsed[name][i]['description'] = desc
                    elif field_type=='Institution':
                        institution = values[k].text
                        dict_parsed[name][i]['institution'] = institution
                    elif field_type=='Source':
                        source = values[k].text
                        dict_parsed[name][i]['source'] = source
                    elif field_type=='Issue':
                        issue = values[k].text
                        dict_parsed[name][i]['issue'] = issue
                    elif field_type=='Report number':
                        report_number = values[k].text
                        dict_parsed[name][i]['report_number'] = report_number
                    elif field_type=='Total citations':
                        list_years = values[k].find_all('span',class_="gsc_oci_g_t")
                        list_years.reverse()
                        num_years = len(list_years)
                        list_values = values[k].find_all('a',class_="gsc_oci_g_a")
                        list_values.reverse() # reversed, so that z-index follows order
                        list_res = []
                        l = 0
                        m = 0
                        while l < num_years:
                            ind = int(list_values[m]['style'].split('z-index:')[1])
                            if l+1 == ind:
                                list_res.append((list_years[l].text,int(list_values[m].text)))
                                m += 1
                            else:
                                list_res.append((list_years[l].text,0))
                            l += 1
                        dict_parsed[name][i]['citations_years'] = list_res
                        dict_parsed[name][i]['citations'] = int(values[k].find('a').text.strip('Cited by '))
    return dict_parsed

def clean_text1(text):
    text1 = re.sub("[^a-zA-Z\s]+", " ", text)
    text1 = re.sub(r'\s+',' ',text1)
    text1 = text1.strip()
    text1 = text1.lower()
    return text1

def top3_bool(val):
    top3 = val.nlargest(3)  # Get the top 3 values in the column
    top3 = top3[top3 > 0] 
    return val.isin(top3)

def map_date(val):
    try:
        return int(val[:4])
    except:
        return np.nan
    
def import_pickle(file):
    return pickle.load(open(file,'rb'))

def map_topic(text):
    dict_main_topics = {'Hardware & Embedded Systems':['Computer Hardware Design','Computing Systems','Signal Processing','Remote Sensing','Robotics','Microelectronics & Electronic Packaging'], 
                    'Cyber Security and Forensics':['Computer Security & Cryptography'], 
                    'Data Management & Analytics':['Data Mining & Analysis','Databases & Information Systems','Library & Information Science'], 
                    'Computational Intelligence':['Game Theory and Decision Science','Fuzzy Systems','Evolutionary Computation','Theoretical Computer Science','Technology Law','Computational Linguistics','Educational Technology'], 
                    'Computer Vision & Language':['Multimedia','Computer Graphics','Human Computer Interaction','Computer Vision & Pattern Recognition'], 
                    'Computer Networks & Communications':['Computer Networks & Wireless Communication','Software Systems'], 
                    'Biomedical Informatics':['Medical Informatics','Bioinformatics & Computational Biology', 'Biomedical Technology', 'Biotechnology'], 
                    'Artificial Intelligence':['Artificial Intelligence','Automation & Control Theory','Sustainable Energy'],
                    'Engineering & Computer Science (general)':['Engineering & Computer Science (general)'],
                    'Others':['Radar, Positioning & Navigation','Nanotechnology', 'Ocean & Marine Engineering', 'Oil, Petroleum & Natural Gas', 'Operations Research', 'Plasma & Fusion', 'Power Engineering', 'Quality & Reliability','Metallurgy', 
                              'Mining & Mineral Resources','Manufacturing & Machinery', 'Materials Engineering', 'Mechanical Engineering','Food Science & Technology',
                              'Environmental & Geological Engineering','Architecture','Aviation & Aerospace Engineering','Ceramic Engineering','Civil Engineering','Combustion & Propulsion',
                             'Transportation', 'Water Supply & Treatment', 'Wood Science & Technology','Textile Engineering','Structural Engineering','Others','Unknown']}
    for area, v in dict_main_topics.items():
        if text in v:
            return area