import streamlit as st
from PIL import Image
import plotly.express as px
import numpy as np
from streamlit_pills import pills
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
from data_transform import parse_gs_publ, clean_text1, top3_bool, map_date, import_pickle, map_topic
import base64
from pyvis.network import Network
import networkx as nx
import streamlit.components.v1 as components
import ast
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from raceplotly.plots import barplot
    
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

#home, listings, budget, prediction= st.tabs(["🏠 Home "," 🗒 Listings "," 💸 Budget "," 📈 Prediction "])
#border: 1px solid #999; border-radius: 4px 4px 0 0;
def tabs(default_tabs = [], default_active_tab=0):
    if not default_tabs:
        return None
    active_tab = st.radio("", default_tabs, index=default_active_tab)
    child = default_tabs.index(active_tab)+1
    st.markdown("""  
        <style type="text/css">
        div[role=radiogroup] > label > div:first-of-type { 
            display: none;
            gap: 0px
        }
        div[role=radiogroup] {
            flex-direction: row
        }
        div[role=radiogroup] label {      
            background: #FFF;
            padding: 6px 14px 8px 8px;
            position: static;
            top: 0px; left: -20px;
            margin-top: -20px;
            border-bottom: 1px solid transparent;
            border-radius: 0 0 0 0
            }
        div[role=radiogroup] label:nth-child(""" + str(child) + """) {   
            background: #EEE !important;
            border-bottom: 1px solid;
        }            
        </style>
    """,unsafe_allow_html=True)        
    return active_tab

#st.header(":rainbow[SCSE Dashboard Playground]")

active_tab = tabs([":rainbow[SCSE Overview]", ":rainbow[Individual Researcher Profile]",":rainbow[SCSE Network]"])
#st.write(active_tab)

@st.cache_data()
def load_data():
    data = pd.read_csv("Part1_data_gs.csv")
    #dict_parsed = parse_gs_publ(data)
    names = sorted(data['Name'].tolist())
    wc_desc = pd.read_csv('wordcloud_text.csv')
    wc_desc.Year = wc_desc['Year'].astype(str)
    #df_parsed = pd.DataFrame()
    #for name in names:
    #    df1 = pd.DataFrame(dict_parsed[name]).T
    #    df1['Name'] = name
    df_parsed = import_pickle('parsed_publications.pkl')
    df_parsed['title_cleaned'] = df_parsed['title'].apply(clean_text1)
    df_parsed['Year'] = df_parsed['date'].apply(map_date)
    df_parsed['Main Topic'] = df_parsed['Subtopic'].apply(map_topic)
    paper_author_groups = df_parsed.groupby('title_cleaned')['Name'].apply(list).reset_index()
    list_authors = []
    num_authors = []
    for _, row in paper_author_groups.iterrows():
        authors = row['Name']
        authors.sort()  # Sort to ensure consistent group assignments
        authors_key = ', '.join(authors)
        num_authors.append(len(authors))
        list_authors.append(authors_key)
    paper_author_groups['authors_key'] = list_authors
    paper_author_groups['num_authors'] = num_authors
    paper_author_groups.drop(0,inplace=True)
    n = len(names)
    my_adj = np.zeros((n,n), dtype= int) # matrix (n,n)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            my_adj[i][j] = sum([True for seq in list_authors if a in seq and b in seq])
    df_adj = pd.DataFrame(my_adj)
    df_adj.index = names
    df_adj.columns = names
    df_adj1 = df_adj.copy()
    for i in range(len(df_adj1)):
        for j in range(len(df_adj1)):
            if df_adj1.iloc[i,j]>0:
                df_adj1.iloc[i,j] = 1
            if i==j:
                df_adj1.iloc[i,j] = 0
    df_adj2 = df_adj.copy()
    for i in range(len(df_adj2)):
        for j in range(len(df_adj2)):
            if i==j:
                df_adj2.iloc[i,j] = 0
    df_sum = df_adj2.sum()
    df_sum1 = df_sum.to_frame().reset_index()
    df_sum1.columns = ['Name','Collaborations']
    top5 = df_sum1.sort_values('Collaborations',ascending=False).head()['Name'].values.tolist()
    df_adj_bool = df_adj2.apply(top3_bool)
    external_coauthors = pd.read_csv('external_coauthors.csv')
    return data, df_parsed, df_adj, df_adj1, df_adj2, df_sum1, df_adj_bool, top5, wc_desc, external_coauthors

df, df_parsed, df_adj, df_adj1, df_adj2, df_sum, df_adj_bool, top5, wc_desc, external_coauthors = load_data()

def ReadPictureFile(wch_fl):
    try:
        return base64.b64encode(open(wch_fl, 'rb').read()).decode()
    except:
        return ""

#@st.cache_data()
def main_page():
    image = Image.open("SCSE_Logo.png")
    st.sidebar.image(image)

    st.sidebar.title('Filter')
    df_count_type = df_parsed.groupby(['Name','Year','type','Main Topic','Subtopic']).agg({'title': 'count','citations':'sum'}).reset_index()
    df_count_type.columns = ['Name','Year','Type','Main Topic','Subtopic','No. of Publications','No. of Citations']
    # Define the available options
    label_year = "Year"
    df_count_type['Year'] = df_count_type['Year'].astype(int)
    list_years = sorted(df_count_type['Year'].unique())
    min_year, max_year = st.sidebar.select_slider(label=label_year,options=list_years,value=(min(list_years),max(list_years)))
    df_count_type = df_count_type[(df_count_type['Year']>=min_year)&(df_count_type['Year']<max_year)]
    
    maintopics = sorted(df_count_type['Main Topic'].unique())
    list_main_area = ['All']+maintopics
    selected_main_area =  st.sidebar.multiselect("Select Main Research Area",list_main_area,default='All',placeholder="Select one or more research areas")
    if 'All' not in selected_main_area:
        df_count_type = df_count_type[df_count_type['Main Topic'].isin(selected_main_area)]
    
    subtopics = sorted(df_count_type['Subtopic'].unique())
    list_area = ['All']+subtopics
    selected_area =  st.sidebar.multiselect("Select Sub Research Area",list_area,default='All',placeholder="Select one or more research areas")
    if 'All' not in selected_area:
        df_count_type = df_count_type[df_count_type['Subtopic'].isin(selected_area)]
    selected_type = pills("Label", ["All", "Journal", "Conference",'Book','Patent','Others'], ["💡","🍀", "🎈", "🗒 ",'💸',"🌈"],label_visibility="collapsed")
    if selected_type != 'All':
        df_parsed2 = df_count_type[df_count_type['Type']==selected_type]
        df_parsed1 = df_parsed2.groupby(['Name']).agg({'No. of Publications': 'sum','No. of Citations':'sum'}).reset_index()
    else:
        df_parsed1 = df_count_type.groupby(['Name']).agg({'No. of Publications': 'sum','No. of Citations':'sum'}).reset_index()
    #else:
    #    df_parsed2 = df_parsed[df_parsed['type']==selected_type]
    #    df_parsed1 = df_parsed2.groupby(['Name']).agg({'title': 'count','citations':'sum'}).reset_index()
    #df_parsed1 = df_count_type.groupby(['Name']).agg({'No. of Publications': 'sum','No. of Citations':'sum'}).reset_index()
    df_parsed1.columns = ['Name','No. of Publications','No. of Citations']
    df_parsed1.sort_values('No. of Publications',ascending=False,inplace=True)
    #df_parsed1 = df_count_type
    publ_container = st.container()
    col1 ,col2 = publ_container.columns(2)
    #if len(selected_rows)>0:
    #    selected_df = pd.DataFrame(selected_rows)
    #    ag1 = AgGrid(selected_df[['Name','No. of Publications','No. of Citations']], enable_enterprise_modules=True)
        #AgGrid(selected_df[['Name','No. of Publications','No. of Citations','image_path']], gridOptions=gridOptions, enable_enterprise_modules=True,allow_unsafe_jscode=True)
    df_count_sub = df_count_type.groupby(['Subtopic','Year'])['No. of Publications'].sum().reset_index()
    #AgGrid(df_count_sub)heatmap_data = df.pivot(index='Subtopic', columns='Year', values='Count').fillna(0)
    df_count_sub['Year'] = df_count_sub['Year'].astype(int)
    with col1:
        df_count_type1 = df_count_type.groupby(['Type'])['No. of Publications'].sum().reset_index()
        fig = px.bar(df_count_type1,x='Type',y='No. of Publications',title="Number of Publications of each type")
        fig["data"][0]["marker"]["color"] = [px.colors.qualitative.Vivid[9] if c == selected_type else px.colors.qualitative.G10[5] for c in fig["data"][0]["x"]]
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
    with col2:
        st.markdown('###### Top 10 Researchers with most Publications')
        df_parsed4 = df_parsed1.head(10)
        gb = GridOptionsBuilder.from_dataframe(df_parsed4)
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, flex=1,enableRowGroup=True, aggFunc="sum", editable=True, sortable=True)
        gridOptions = gb.build()
        grid_table = AgGrid(df_parsed4, gridOptions=gridOptions, enable_enterprise_modules=True,allow_unsafe_jscode=True)
        
    citations_container = st.container()
    col3, col4 = citations_container.columns(2)
    with col3:
        df_count_type1 = df_count_type.groupby(['Type'])['No. of Citations'].sum().reset_index()
        fig = px.bar(df_count_type1,x='Type',y='No. of Citations',title="Number of Citations of each type")
        fig["data"][0]["marker"]["color"] = [px.colors.qualitative.Vivid[9] if c == selected_type else px.colors.qualitative.G10[5] for c in fig["data"][0]["x"]]
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col4:
        st.markdown('###### Top 10 Researchers with most Citations')
        df_parsed3 = df_parsed1.sort_values(by='No. of Citations', ascending=False)
        df_parsed3 = df_parsed3.head(10)
        gb1 = GridOptionsBuilder.from_dataframe(df_parsed3)
        gb1.configure_side_bar()
        gb1.configure_default_column(flex=1,groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True, sortable=True)
        gridOptions1 = gb1.build()
        grid_table1 = AgGrid(df_parsed3, gridOptions=gridOptions1, enable_enterprise_modules=True,allow_unsafe_jscode=True)
        
    topic_container = st.container()
    col5,col6 = topic_container.columns(2)
    with col5:
        df_count_main = df_count_type.groupby(['Main Topic']).agg({'No. of Publications':'sum','No. of Citations':'sum'}).reset_index()
        fig3 = px.bar(df_count_main,x='No. of Publications',y='Main Topic',orientation='h',title='No. of Publications in each Topic')
        fig3["data"][0]["marker"]["color"] = [px.colors.qualitative.G10[5] for c in fig3["data"][0]["x"]]
        st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
    with col6:
        st.markdown('###### Number of Publications in each Research Area')
        AgGrid(df_count_main)
        
    df_count_sub.sort_values(by='Year',inplace=True)
    #fig.update_xaxes(range=[0,30])
    #fig.update_yaxes(range=[0,50])
    chart_placeholder = st.empty()
    years1 = list(df_count_sub['Year'].unique())
    selected_year = st.select_slider('Year',options=years1,label_visibility="collapsed")
    if selected_year in years1:
        df_count_sub1 = df_count_sub[df_count_sub['Year']==selected_year]
        fig5 = px.bar(df_count_sub1, x="No. of Publications", y="Subtopic", orientation='h',title='Change in Research topics over the years')
        y_labels = df_count_sub1['Subtopic'].unique()
        fig5.update_layout(yaxis=dict(
        tickvals=list(range(len(y_labels))),
        ticktext=y_labels
        ))
        fig5["data"][0]["marker"]["color"] = [px.colors.qualitative.G10[5] for c in fig5["data"][0]["x"]]
        chart_placeholder.plotly_chart(fig5, theme="streamlit",use_container_width=True)
        
    

    
def individual_page():
    image = Image.open("SCSE_Logo.png")
    st.sidebar.image(image)
    avg_all = df_parsed.groupby(['Name','Year','type','Subtopic']).agg({'title': 'count','citations':'sum'}).reset_index()
    st.sidebar.title('Filter')
    label = "Research Profile"
    list_researchers = sorted(df['Name'].tolist())
    researcher = st.sidebar.selectbox(label,list_researchers,index=None,placeholder='Select Researcher')
    #st.sidebar.markdown('')
    
    if researcher != None:
        #AgGrid(df)
        df_prof1 = df_parsed[df_parsed['Name']==researcher]
        label_year = "Year"
        years = [int(i) for i in set(df_prof1['Year'].values) if np.isnan(i)==False]
        list_years = sorted(years)
        min_year, max_year = st.sidebar.select_slider(label=label_year,options=list_years,value=(min(years),max(years)))
        df_prof = df_prof1[(df_prof1['Year']>=min_year)&(df_prof1['Year']<=max_year)]
        avg_all = avg_all[(avg_all['Year']>=min_year)&(avg_all['Year']<=max_year)]
        df_prof.reset_index(drop=True,inplace=True)
        #st.sidebar.markdown('')
        maintopics = sorted(list(set(df_prof['Main Topic'].values)))
        list_mainarea = ['All']+maintopics
        selected_mainarea =  st.sidebar.multiselect("Select Main Research Area",list_mainarea,default='All',placeholder="Select one or more areas")
        if 'All' not in selected_mainarea:
            avg_all = avg_all[avg_all['Main Topic'].isin(selected_mainarea)]
            df_prof = df_prof[df_prof['Main Topic'].isin(selected_mainarea)]
            df_prof.reset_index(drop=True,inplace=True)
            
        subtopics = sorted(list(set(df_prof['Subtopic'].values)))
        list_area = ['All']+subtopics
        selected_area =  st.sidebar.multiselect("Select Subtopics",list_area,default='All',placeholder="Select one or more areas")
        if 'All' not in selected_area:
            avg_all = avg_all[avg_all['Subtopic'].isin(selected_area)]
            df_prof = df_prof[df_prof['Subtopic'].isin(selected_area)]
            df_prof.reset_index(drop=True,inplace=True)
        list_type = ['All','📕 Journal','📗 Conference','📘 Book','📙 Patent','📚 Others']
        selected_type = st.sidebar.multiselect("Select Type",list_type,default='All',placeholder='Select one or more Types')
        if 'All' not in selected_type:
            selected_types = [i[2:] for i in selected_type]
            df_prof = df_prof[df_prof['type'].isin(selected_types)]
            df_prof.reset_index(drop=True,inplace=True)
            avg_all = avg_all[avg_all['type'].isin(selected_types)]
        label_sort = 'Sort Publications by'
        list_sort = ['Citations','Year']
        selected_sort = st.sidebar.selectbox(label_sort,list_sort,index=1,placeholder='Sort by')
        
        label_venue = 'Venues'
        venue_expander = st.sidebar.expander(label_venue)
        with venue_expander:
            list_venues = list(df_prof[df_prof['Name']==researcher]['venue_all'].values)
            dict_count = Counter(list_venues).most_common()
            for venue, count in dict_count:
                grade = df_prof[(df_prof['Name']==researcher)&(df_prof['venue_all']==venue)]['Rankings'].values[0]
                st.markdown(f'({count}) {venue}\n (Venue Quality: {grade})')
                
        col1, mid, col2 = st.columns([1,5,20])
        with col1:
            st.image('images/'+researcher+'.jpg', width=180)
        with col2:
            st.markdown(f'#### Researcher: {researcher}')
            position = df[df['Name']==researcher]['Position'].values[0]
            list_position = ast.literal_eval(position)
            for pos in list_position:
                st.markdown(pos)
            button1, button2, button3, _, _,_ = st.columns([1,1,1,1,1,1])
            with button1:
                email = df[df['Name']==researcher]['Email'].values[0]
                st.link_button('📧 Email',email)
            with button2:
                drntu = df[df['Name']==researcher]['DRNTU'].values[0]
                st.link_button('🔗 DRNTU',drntu)
            with button3:
                try:
                    website = df[df['Name']==researcher]['Website'].values[0]
                    st.link_button('✉️ Website',website)
                except:
                    pass
            ini_list = df[df['Name']==researcher]['Author Keywords'].values[0]
            list_keywords = ast.literal_eval(ini_list)
            if len(list_keywords)>0:
                pills('Author Keywords',list_keywords,label_visibility="collapsed",index=None)
        st.markdown('')
        bio_tab, publ_tab, ins_tab = st.tabs([' Profile ',' Publications ',' Insights '])
        st.markdown("""
        <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
                display: inline-flex;
                word-wrap: break-word;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size:1.5rem;
                color: black;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #FFFFFF;
                border-radius: 0px 0px 0px 0px;
                gap: 10px;
                padding: 6px 12px 8px 8px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #EEEEEE !important;
                color: black;
                border-style: none;
                gap: 10px;
            }
        </style>""", unsafe_allow_html=True)
                
        with bio_tab:
            st.markdown('##### Biography')
            bio_text = df[df['Name']==researcher]['Biography_parsed'].values[0]
            bio_text = bio_text.replace('\n\n','\n')
            bio_text_ = bio_text.split('Sections:')
            bio_text1 = bio_text_[0].replace('Timeline:','')
            bio_text2 = bio_text_[1]
            st.markdown(bio_text1)
            st.markdown(bio_text2)
            st.markdown('##### Research Interests')
            ini_list = df[df['Name']==researcher]['Research Area'].values[0]
            list_research_areas = ast.literal_eval(ini_list)
            if len(list_research_areas)>0:
                pills('Research Interests',list_research_areas, label_visibility="collapsed",index=None)
            ini_list = df[df['Name']==researcher]['Research Interest'].values[0]
            list_research_interest = ast.literal_eval(ini_list)
            if len(list_research_interest)>0:
                st.write('- '+'\n\n - '.join(list_research_interest))
        with publ_tab:    
            #AgGrid(df_prof)
            if selected_sort=='Year':
                df_year = df_prof.sort_values(by=['Year','citations'],ascending=False)
                list_years = sorted(list(set(df_year['Year'].values.tolist())),reverse=True)
                for year in list_years:
                    if np.isnan(year)==False:
                        st.markdown('#### '+str(year)[:-2])
                        df_year1 = df_year[df_year['Year']==year]
                        df_year1.reset_index(drop=True,inplace=True)
                        for i in range(len(df_year1)):
                            title = df_year1['title'][i]
                            authors = df_year1['authors'][i]
                            type = df_year1['type'][i]
                            citations = df_year1['citations'][i]
                            if str(citations)=='nan':
                                citations = 0
                            if type=='Journal':
                                journal = df_year1['journal'][i]
                                st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📕 {journal}''', unsafe_allow_html=True)
                            elif type=='Conference':
                                conf = df_year1['conference'][i]
                                st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📗 {conf}''', unsafe_allow_html=True)
                            elif type=='Book':
                                book = df_year1['book'][i]
                                st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📘 {book}''', unsafe_allow_html=True)
                            elif type=='Patent':
                                patent = df_year1['patent'][i]
                                st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📙 {patent} Patent''', unsafe_allow_html=True)
                            elif type=='Others':
                                source = df_year1['source'][i]
                                publisher = df_year1['publisher'][i]
                                if str(source)!='nan':
                                    st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📚 {source}''', unsafe_allow_html=True)
                                elif str(publisher)!= 'nan':
                                    st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📚 {publisher}''', unsafe_allow_html=True)
                            with st.expander('See more details'):
                                st.markdown('**Description**')
                                abstract = df_year1['description'][i]
                                st.markdown(abstract)
                                if str(df_year1['citations_years'][i])!='nan':
                                    df_citations_val = pd.DataFrame(df_year1['citations_years'][i])
                                    df_citations_val.columns = ['Year','Value']
                                    fig = px.bar(df_citations_val,x='Year',y='Value',title="Number of Citations over the years")
                                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            elif selected_sort=='Citations':
                df_citations_new = df_prof.sort_values(by=['citations','Year'],ascending=False)
                for i in range(len(df_citations_new)):
                    title = df_citations_new['title'][i]
                    authors = df_citations_new['authors'][i]
                    type = df_citations_new['type'][i]
                    citations = df_citations_new['citations'][i]
                    year = str(df_citations_new['Year'][i])[:-2]
                    if str(citations)=='nan':
                        citations = 0
                    if type=='Journal':
                        journal = df_citations_new['journal'][i]
                        st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📕 {journal} **{year}**''', unsafe_allow_html=True)
                    elif type=='Conference':
                        conf = df_citations_new['conference'][i]
                        st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📗 {conf} **{year}**''', unsafe_allow_html=True)
                    elif type=='Book':
                        book = df_citations_new['book'][i]
                        st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📘 {book} **{year}**''', unsafe_allow_html=True)
                    elif type=='Patent':
                        patent = df_citations_new['patent'][i]
                        st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📙 {patent} Patent **{year}**''', unsafe_allow_html=True)
                    elif type=='Others':
                        source = df_citations_new['source'][i]
                        publisher = df_citations_new['publisher'][i]
                        if str(source)!='nan':
                            st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📚 {source} **{year}**''', unsafe_allow_html=True)
                        elif str(publisher)!= 'nan':
                            st.markdown(f'''- **{title}**: Cited by {citations}<br> {authors}<br>📚 {publisher} **{year}**''', unsafe_allow_html=True)
                    with st.expander('See more details'):
                        st.markdown('**Description**')
                        abstract = df_citations_new['description'][i]
                        st.markdown(abstract)
                        if str(df_citations_new['citations_years'][i])!='nan':
                            df_citations_val = pd.DataFrame(df_citations_new['citations_years'][i])
                            df_citations_val.columns = ['Year','Value']
                            fig = px.bar(df_citations_val,x='Year',y='Value',title="Number of Citations over the years")
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            st.markdown('Note: 📕 refers to Journals,📗 refers to Conferences,📘 refers to Books, 📙 refers to Patents, 📚 refers to Other Publications.')
                    
        with ins_tab:
            wc_desc1 = wc_desc[wc_desc['Name']==researcher]
            wc_desc1 = wc_desc1.groupby('words').agg({'count':'sum'}).reset_index()
            data = wc_desc1.sort_values(by='count',ascending=False)
            data = data[:20]
            #AgGrid(avg_all)
            avg_all_name = avg_all.groupby('Name').agg({'title':'sum','citations':'sum'}).reset_index()
            try:
                avg_publications = avg_all_name['title'].mean()
                avg_citations = avg_all_name['citations'].mean()
                no_publ = avg_all_name[avg_all_name['Name']==researcher]['title'].values[0]
                no_cit = avg_all_name[avg_all_name['Name']==researcher]['citations'].values[0]
                diff_publ = no_publ-avg_publications
                diff_cit = no_cit-avg_citations
                col1_metric, col2_metric, col3_metric = st.columns(3)
                col1_metric.metric("Publications", no_publ, round(diff_publ,0))
                col2_metric.metric("Citations", no_cit, round(diff_cit,0))
                try:
                    collab_ind = df.index[df['Name']==researcher]
                    collab = df['Frequent_collab'][collab_ind].values[0]
                    col3_metric.markdown('Frequent Collaborators')
                    col3_metric.markdown('##### '+collab)
                except:
                    pass
                #col3_metric.metric("Humidity", "86%", "4%")
                df_count_publ = df_prof.groupby('Year').agg({'title':'count'}).reset_index()
                df_count_publ.columns = ['Year','Count']
                fig = px.bar(df_count_publ,x='Year',y='Count',title="Number of Publications over the years")
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                df_citations = pd.DataFrame()
                for i in range(len(df_prof)):
                    if str(df_prof['citations_years'][i])!='nan':
                        df_citations1 = pd.DataFrame(df_prof['citations_years'][i])
                        df_citations1.columns = ['Year','Value']
                        df_citations = pd.concat([df_citations,df_citations1],ignore_index=True)
                df_citations['Year'] = df_citations['Year'].astype(int)
                df_citations = df_citations[(df_citations['Year']<=max_year)&(df_citations['Year']>=min_year)]
                fig = px.bar(df_citations,x='Year',y='Value',title="Number of Citations over the years")
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                st.markdown('')
                
                # Venue type & quality
                venue_col1, venue_col2 = st.columns(2)
                with venue_col1:
                    list_types = list(df_prof[df_prof['Name']==researcher]['type'].values)
                    dict_types = Counter(list_types)
                    myKeys = list(dict_types.keys())
                    myKeys.sort()
                    sorted_dict = {i: dict_types[i] for i in myKeys}
                    fig_type = px.bar(x=sorted_dict.keys(),y=sorted_dict.values(),labels=dict(x='Type',y='No. of Publications'),title='No. of Publications of each type')
                    st.plotly_chart(fig_type, theme="streamlit", use_container_width=True)
                with venue_col2:
                    list_ranks = list(df_prof[df_prof['Name']==researcher]['Rankings'].values)
                    dict_ranks = Counter(list_ranks)
                    myKeys_rank = list(dict_ranks.keys())
                    myKeys_rank.sort()
                    sorted_dict_rank = {i: dict_ranks[i] for i in myKeys_rank}
                    fig_ranks = px.bar(x=sorted_dict_rank.keys(),y=sorted_dict_rank.values(),labels=dict(x='Venue Quality',y='No. of Publications'),title='No. of Publications over Venue Quality')
                    st.plotly_chart(fig_ranks, theme="streamlit", use_container_width=True)
                
                col5_topic,col6_topic = st.columns(2)
                df_count_main = df_prof.groupby(['Main Topic']).agg({'title':'count','citations':'sum'}).reset_index()
                df_count_main.columns = ['Main Topic','No. of Publications','No. of Citations']
                df_count_sub = df_prof.groupby(['Subtopic']).agg({'title':'count','citations':'sum'}).reset_index()
                df_count_sub.columns = ['Subtopic','No. of Publications','No. of Citations']
                with col5_topic:
                    fig_p = px.bar(df_count_main,x='No. of Publications',y='Main Topic',orientation='h',title='No. of Publications in each Topic')
                    st.plotly_chart(fig_p, theme="streamlit", use_container_width=True)
                with col6_topic:
                    fig_c = px.bar(df_count_main,x='No. of Citations',y='Main Topic',orientation='h',title='No. of Citations in each Topic')
                    st.plotly_chart(fig_c, theme="streamlit", use_container_width=True)
                
                df_count_sub_year = df_prof.groupby(['Year','Subtopic']).agg({'title':'count','citations':'sum'}).reset_index()
                df_count_sub_year.columns = ['Year','Subtopic','No. of Publications','No. of Citations']
                df_count_sub_year.sort_values(by='Year',inplace=True)
                df_count_sub_year_pivot = df_count_sub_year.pivot(index='Year', columns='Subtopic', values='No. of Publications').fillna(0).reset_index()
                df_count_sub_year1 = pd.DataFrame()
                for col in df_count_sub_year_pivot.columns:
                    if col!='Year':
                        df_random = df_count_sub_year_pivot[['Year',col]]
                        df_random['Subtopic'] = col
                        df_random.columns = ['Year','No. of Publications','Subtopic']
                        df_count_sub_year1 = pd.concat([df_count_sub_year1,df_random],ignore_index=True)
                df_count_sub_year1.sort_values(by=['Year','Subtopic'],inplace=True)
                my_raceplot = barplot(df_count_sub_year1,
                    item_column='Subtopic',
                    value_column='No. of Publications',
                    time_column='Year')

                fig = my_raceplot.plot(title = 'Change in Research topics over the years',
                                item_label = 'Subtopics',
                                value_label = 'No. of Publications',
                                frame_duration = 2000)
                fig.update_layout(autosize=False,height=700)
                #fig.update_traces(width=0.5)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('##### Most frequently used words in Abstracts')
                #bar_col, content_col = wc_container.columns([2, 4], gap="large")
                st.markdown('')
                bar_tab, tree_tab, wc_tab = st.tabs([' Bar Plot ',' Treemap ',' Wordcloud '])
                st.markdown("""
                <style>
                    .stTabs [data-baseweb="tab-list"] {
                        gap: 1px;
                        display: inline-flex;
                        word-wrap: break-word;
                    }
                    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                        font-size:1rem;
                        color: black;
                    }
                    .stTabs [data-baseweb="tab"] {
                        background-color: #FFFFFF;
                        border-radius: 0px 0px 0px 0px;
                        gap: 1px;
                        padding: 6px 12px 8px 8px;
                    }
                    .stTabs [aria-selected="true"] {
                        background-color: #EEEEEE !important;
                        color: black;
                        border-style: none;
                        gap: 1px;
                    }
                </style>""", unsafe_allow_html=True)
                with bar_tab:
                    fig=px.bar(data,x='count',y='words', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                with tree_tab:
                    wc_desc1 = wc_desc[wc_desc['Name']==researcher]
                    wc_desc1 = wc_desc1.groupby('Year').head(10).reset_index(drop=True)
                    fig1 = px.treemap(wc_desc1, path=[px.Constant("Words"), 'Year','words'],
                    values='count',
                    color='count',
                    color_continuous_scale='viridis',
                    color_continuous_midpoint=np.average(wc_desc1['count'])
                    )
                    fig1.update_layout(margin = dict(t=50, l=30, r=30, b=20))
                    fig1.update_traces(sort=False, selector=dict(type='treemap'))
                    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
                with wc_tab:
                    wc_desc1 = wc_desc[wc_desc['Name']==researcher]
                    wc_desc1 = wc_desc1.groupby('words').agg({'count':'sum'}).reset_index()
                    word_frequency = dict(zip(wc_desc1['words'].values,wc_desc1['count'].values))
                    WC_height = 1000
                    WC_width = 2000
                    WC_max_words = 100
                    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,background_color="white")
                    wordCloud.generate_from_frequencies(word_frequency)
                    plt.imshow(wordCloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot()
            except:
                pass
                

def network_page():
    #AgGrid(df)
    image = Image.open("SCSE_Logo.png")
    st.sidebar.image(image)
    community = st.sidebar.selectbox('Community',['NTU SCSE Community','External Community'],index=None)
    G1 = nx.from_pandas_adjacency(df_adj)
    #G = Network(height='1000px', bgcolor='#222222', font_color='white',notebook=True)
    if community=='NTU SCSE Community':
        st.markdown("#### NTU SCSE Community")
        G = Network(
            notebook=False,
            cdn_resources="remote",
            bgcolor="white",
            font_color="black",
            height="750px",
            width="100%",
            select_menu=True,
            filter_menu=True,
        )
        nodes = G1.nodes
        for node in nodes:
            index = df.index[df['Name']==node]
            #img = Image.open('images/'+node+'.jpg')
            #Image.open('images/'+node+'.jpg')
            #G.add_node(node,label=node,shape='circularImage',image=np.array(img),borderWidth=3)
            G.add_node(node,label=node,shape='circularImage',image=df['image_path'][index].values[0],borderWidth=3)
        for node in nodes:
            for node1 in nodes:
                if (df_adj_bool[node][node1]==True) and (node != node1):
                    G.add_edge(node,node1,title=str(df_adj[node][node1]),color='#FF4E50')
                elif (df_adj[node][node1]>0) and (node != node1):
                    G.add_edge(node,node1,title=str(df_adj[node][node1]),color='#45ADA8')
        for n in G.nodes:
            if n['label'] in top5:
                n['color'] = '#E74C3C'
            name = n['label']
            iposition = df.index[df['Name']==name]
            lposition = df['Position'][iposition].values[0]
            position = ast.literal_eval(lposition)[0]
            publications = df['Publications'][iposition].values[0]
            citations = df['Citations'][iposition].values[0]
            freq = df['Frequent_collab'][iposition].values[0]
            if str(freq)!='nan':
                if len(freq.split(', '))>5:
                    freq_1 = ', '.join(freq.split(', ')[:5])
                    freq_2 = ', '.join(freq.split(', ')[5:])
                    freq = freq_1+',\n'+freq_2
            n['title'] = f'Name: {name}\nPosition: {position}\nPublications: {publications}\nCitations: {citations}\nFrequent collaborators: {freq}'
            
        G.repulsion(node_distance=100, spring_length=400)
        G.inherit_edge_colors(False)
        G.save_graph('network.html')
        HtmlFile = open("network.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height = 1200,width=1000)
    
    elif community=='External Community':
        st.markdown("#### External Community")
        list_profs = external_coauthors['index'].unique()
        selected_prof = st.sidebar.selectbox('Select Professor',list_profs,index=None)
        df_prof = external_coauthors[external_coauthors['index']==selected_prof]
        df_prof.reset_index(drop=True,inplace=True)
        if selected_prof != None:
            try:
                G_ext = Network(notebook=False,cdn_resources="remote",bgcolor="white",font_color="black",height="750px",
                    width="100%",select_menu=True,filter_menu=True)
                ind = df.index[df['Name']==selected_prof]
                img_path = df['image_path'][ind].values[0]
                lposition = df['Position'][ind].values[0]
                position = ast.literal_eval(lposition)[0]
                publications = df['Publications'][ind].values[0]
                citations = df['Citations'][ind].values[0]
                freq = df['Frequent_collab'][ind].values[0]
                if str(freq)!='nan':
                    if len(freq.split(', '))>5:
                        freq_1 = ', '.join(freq.split(', ')[:5])
                        freq_2 = ', '.join(freq.split(', ')[5:])
                        freq = freq_1+',\n'+freq_2
                title = f'Name: {selected_prof}\nPosition: {position}\nPublications: {publications}\nCitations: {citations}\nFrequent collaborators in NTU: {freq}'
                G_ext.add_node(selected_prof,label=selected_prof,shape='circularImage',image=img_path,title=title)
                for val in df_prof['Name'].values:
                    iposition = df_prof.index[df_prof['Name']==val]
                    try:
                        aff = df_prof['Affiliation'][iposition].values[0]
                    except:
                        aff = ''
                    try:
                        ver = df_prof['Verified'][iposition].values[0]
                    except:
                        ver = ''
                    title = f'Name: {val}\nAffiliation: {aff}\n{ver}'
                    if ('ntu.' in title.lower()) or ('nanyang technological university' in title.lower()):
                        col = '#FF4E50'
                    else:
                        col = '#45ADA8'
                    G_ext.add_node(val,label=val,title=title,color=col)
                    G_ext.add_edge(selected_prof,val)
                G_ext.save_graph('ext_network.html')
                HtmlFile = open("ext_network.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read() 
                components.html(source_code, height = 1200,width=1000)
                #AgGrid(df_prof[['Name','Affiliation','Verified']])
            except:
                pass

if active_tab==":rainbow[SCSE Overview]":
    main_page()
elif active_tab==":rainbow[Individual Researcher Profile]":
    individual_page()
elif active_tab==":rainbow[SCSE Network]":
    network_page()
    
    
