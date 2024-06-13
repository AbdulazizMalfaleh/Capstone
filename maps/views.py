# views.py
from django.shortcuts import render, redirect
import pandas as pd
import os
import requests
from django.conf import settings


import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import re




































# views.py
from django.shortcuts import render, redirect
import pandas as pd
import os
import requests
from django.conf import settings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

def get_district_name(lat, lng, api_key):
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}&language=ar"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        if results:
            for component in results[0]['address_components']:
                if 'sublocality_level_1' in component['types'] or 'administrative_area_level_2' in component['types']:
                    return component['long_name']
    return "Unknown"

def pre(lat1, lng1, lat2, lng2, budget):
    latitude = lat1
    longitude = lng1
    work_location = [latitude, longitude]
    
    latitude2 = lat2
    longitude2 = lng2
    wife_location = [latitude2, longitude2]
    
    user_price_per_meter = budget
        # Read CSV files from static data directory
    merged_df_path = os.path.join(settings.STATIC_ROOT, 'data', 'merged_df.csv')
    schools_df_path = os.path.join(settings.STATIC_ROOT, 'data', 'final_schools_data.csv')
    mosques_df_path = os.path.join(settings.STATIC_ROOT, 'data', 'mousq_final_data.csv')
    districts_df_path = os.path.join(settings.STATIC_ROOT, 'data', 'final_district_data.csv')

    df = pd.read_csv(merged_df_path)
    schools_df = pd.read_csv(schools_df_path)
    mosques_df = pd.read_csv(mosques_df_path)
    districts_df = pd.read_csv(districts_df_path).drop(columns=['Unnamed: 0'])

    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        r = 6371  
        return c * r

    X_10th = df['10th Percentile']
    X_25th = df['25th Percentile']
    X_mean = df['Mean']
    X_75th = df['75th Percentile']
    X_90th = df['90th Percentile']

    scaler = StandardScaler()
    X_10th = scaler.fit_transform(X_10th.values.reshape(-1, 1))
    X_25th = scaler.fit_transform(X_25th.values.reshape(-1, 1))
    X_mean = scaler.fit_transform(X_mean.values.reshape(-1, 1))
    X_75th = scaler.fit_transform(X_75th.values.reshape(-1, 1))
    X_90th = scaler.fit_transform(X_90th.values.reshape(-1, 1))

    knn_10th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_10th)
    knn_25th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_25th)
    knn_mean = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_mean)
    knn_75th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_75th)
    knn_90th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_90th)

    user_input = [[user_price_per_meter]]
    user_input_scaled = scaler.transform(user_input)

    distances_10th, indices_10th = knn_10th.kneighbors(user_input_scaled)
    candidate_neighborhoods_10th = df.iloc[indices_10th[0]]

    distances_25th, indices_25th = knn_25th.kneighbors(user_input_scaled)
    candidate_neighborhoods_25th = df.iloc[indices_25th[0]]

    distances_mean, indices_mean = knn_mean.kneighbors(user_input_scaled)
    candidate_neighborhoods_mean = df.iloc[indices_mean[0]]

    distances_75th, indices_75th = knn_75th.kneighbors(user_input_scaled)
    candidate_neighborhoods_75th = df.iloc[indices_75th[0]]

    distances_90th, indices_90th = knn_90th.kneighbors(user_input_scaled)
    candidate_neighborhoods_90th = df.iloc[indices_90th[0]]

    candidate_neighborhoods_10th['Distance_to_Work'] = candidate_neighborhoods_10th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_10th['Distance_to_Wife'] = candidate_neighborhoods_10th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_25th['Distance_to_Work'] = candidate_neighborhoods_25th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_25th['Distance_to_Wife'] = candidate_neighborhoods_25th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_mean['Distance_to_Work'] = candidate_neighborhoods_mean.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_mean['Distance_to_Wife'] = candidate_neighborhoods_mean.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_75th['Distance_to_Work'] = candidate_neighborhoods_75th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_75th['Distance_to_Wife'] = candidate_neighborhoods_75th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_90th['Distance_to_Work'] = candidate_neighborhoods_90th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_90th['Distance_to_Wife'] = candidate_neighborhoods_90th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    def count_nearby_schools(lon, lat, schools_df, max_distance=3):
        return sum(haversine(lon, lat, school_lon, school_lat) <= max_distance for school_lat, school_lon in zip(schools_df['latitude'], schools_df['longitude']))

    def count_nearby_mosques(lon, lat, mosques_df, max_distance=3):
        return sum(haversine(lon, lat, mosques_lon, mosques_lat) <= max_distance for mosques_lat, mosques_lon in zip(mosques_df['latitude'], mosques_df['longitude']))

    candidate_neighborhoods_10th['Nearby_Mosques'] = candidate_neighborhoods_10th.apply(
        lambda row: count_nearby_mosques(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_10th['Nearby_Schools'] = candidate_neighborhoods_10th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_25th['Nearby_Mosques'] = candidate_neighborhoods_25th.apply(
        lambda row: count_nearby_mosques(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_25th['Nearby_Schools'] = candidate_neighborhoods_25th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_mean['Nearby_Mosques'] = candidate_neighborhoods_mean.apply(
        lambda row: count_nearby_mosques(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_mean['Nearby_Schools'] = candidate_neighborhoods_mean.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_75th['Nearby_Mosques'] = candidate_neighborhoods_75th.apply(
        lambda row: count_nearby_mosques(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_75th['Nearby_Schools'] = candidate_neighborhoods_75th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_90th['Nearby_Mosques'] = candidate_neighborhoods_90th.apply(
        lambda row: count_nearby_mosques(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_90th['Nearby_Schools'] = candidate_neighborhoods_90th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_10th['Combined_Score'] = (candidate_neighborhoods_10th['Distance_to_Work'] + candidate_neighborhoods_10th['Distance_to_Wife']) / 2
    candidate_neighborhoods_10th = candidate_neighborhoods_10th.sort_values(by='Combined_Score')

    candidate_neighborhoods_25th['Combined_Score'] = (candidate_neighborhoods_25th['Distance_to_Work'] + candidate_neighborhoods_25th['Distance_to_Wife']) / 2
    candidate_neighborhoods_25th = candidate_neighborhoods_25th.sort_values(by='Combined_Score')

    candidate_neighborhoods_mean['Combined_Score'] = (candidate_neighborhoods_mean['Distance_to_Work'] + candidate_neighborhoods_mean['Distance_to_Wife']) / 2
    candidate_neighborhoods_mean = candidate_neighborhoods_mean.sort_values(by='Combined_Score')

    candidate_neighborhoods_75th['Combined_Score'] = (candidate_neighborhoods_75th['Distance_to_Work'] + candidate_neighborhoods_75th['Distance_to_Wife']) / 2
    candidate_neighborhoods_75th = candidate_neighborhoods_75th.sort_values(by='Combined_Score')

    candidate_neighborhoods_90th['Combined_Score'] = (candidate_neighborhoods_90th['Distance_to_Work'] + candidate_neighborhoods_90th['Distance_to_Wife']) / 2
    candidate_neighborhoods_90th = candidate_neighborhoods_90th.sort_values(by='Combined_Score')

    recommended_neighborhoods_10th = candidate_neighborhoods_10th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_25th = candidate_neighborhoods_25th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_mean = candidate_neighborhoods_mean[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_75th = candidate_neighborhoods_75th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_90th = candidate_neighborhoods_90th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]

    df_10th = recommended_neighborhoods_10th.head(2)
    df_25th = recommended_neighborhoods_25th.head(2)
    df_mean = recommended_neighborhoods_mean.head(2)
    df_75th = recommended_neighborhoods_75th.head(2)
    df_90th = recommended_neighborhoods_90th.head(2)

    df_10th['Color'] = 'Red'
    df_10th['percentile'] = '10th'
    df_25th['Color'] = 'Blue'
    df_25th['percentile'] = '25th'
    df_mean['Color'] = 'Green'
    df_mean['percentile'] = 'Mean'
    df_75th['Color'] = 'Yellow'
    df_75th['percentile'] = '75th'
    df_90th['Color'] = 'Orange'
    df_90th['percentile'] = '90th'

    combined_df = pd.concat([df_10th, df_25th, df_mean, df_75th, df_90th])

    combined_df.reset_index(drop=True, inplace=True)
    combined_df = combined_df.drop_duplicates(subset=['District'], keep='first').reset_index(drop=True)
    return combined_df
'''
def pre(lat1, lng1,lat2, lng2,budget):



    latitude =lat1


    longitude =lng1
    work_location = list([latitude,longitude])





    
    latitude2 = lat2

   
    longitude2 = lng2
    wife_location = list([latitude2,longitude2])

    user_price_per_meter=budget

    df = pd.read_csv('Model_Data/merged_df.csv')
    schools_df = pd.read_csv('Model_Data/final_schools_data.csv')
    mosques_df = pd.read_csv('Model_Data/mousq_final_data.csv')
    districts_df = pd.read_csv('Model_Data/final_district_data.csv').drop(columns=['Unnamed: 0'])

    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        r = 6371  
        return c * r

    X_10th = df['10th Percentile']
    X_25th = df['25th Percentile']
    X_mean = df['Mean']
    X_75th = df['75th Percentile']
    X_90th = df['90th Percentile']

    scaler = StandardScaler()
    X_10th = scaler.fit_transform(X_10th.values.reshape(-1, 1))
    X_25th = scaler.fit_transform(X_25th.values.reshape(-1, 1))
    X_mean = scaler.fit_transform(X_mean.values.reshape(-1, 1))
    X_75th = scaler.fit_transform(X_75th.values.reshape(-1, 1))
    X_90th = scaler.fit_transform(X_90th.values.reshape(-1, 1))

    knn_10th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_10th)
    knn_25th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_25th)
    knn_mean = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_mean)
    knn_75th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_75th)
    knn_90th = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_90th)

    user_input = [[user_price_per_meter]]
    user_input_scaled = scaler.transform(user_input)

    distances_10th, indices_10th = knn_10th.kneighbors(user_input_scaled)
    candidate_neighborhoods_10th = df.iloc[indices_10th[0]]

    distances_25th, indices_25th = knn_25th.kneighbors(user_input_scaled)
    candidate_neighborhoods_25th = df.iloc[indices_25th[0]]

    distances_mean, indices_mean = knn_mean.kneighbors(user_input_scaled)
    candidate_neighborhoods_mean = df.iloc[indices_mean[0]]

    distances_75th, indices_75th = knn_75th.kneighbors(user_input_scaled)
    candidate_neighborhoods_75th = df.iloc[indices_75th[0]]

    distances_90th, indices_90th = knn_90th.kneighbors(user_input_scaled)
    candidate_neighborhoods_90th = df.iloc[indices_90th[0]]



    candidate_neighborhoods_10th['Distance_to_Work'] = candidate_neighborhoods_10th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_10th['Distance_to_Wife'] = candidate_neighborhoods_10th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_25th['Distance_to_Work'] = candidate_neighborhoods_25th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_25th['Distance_to_Wife'] = candidate_neighborhoods_25th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_mean['Distance_to_Work'] = candidate_neighborhoods_mean.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_mean['Distance_to_Wife'] = candidate_neighborhoods_mean.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_75th['Distance_to_Work'] = candidate_neighborhoods_75th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_75th['Distance_to_Wife'] = candidate_neighborhoods_75th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    candidate_neighborhoods_90th['Distance_to_Work'] = candidate_neighborhoods_90th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], work_location[1], work_location[0]), axis=1)

    candidate_neighborhoods_90th['Distance_to_Wife'] = candidate_neighborhoods_90th.apply(
        lambda row: haversine(row['longitude'], row['latitude'], wife_location[1], wife_location[0]), axis=1)

    def count_nearby_schools(lon, lat, schools_df, max_distance=3):
        return sum(haversine(lon, lat, school_lon, school_lat) <= max_distance for school_lat, school_lon in zip(schools_df['latitude'], schools_df['longitude']))

    def count_nearby_mosques(lon, lat, mosques_df, max_distance=3):
        return sum(haversine(lon, lat, mosques_lon, mosques_lat) <= max_distance for mosques_lat, mosques_lon in zip(mosques_df['latitude'], mosques_df['longitude']))

    candidate_neighborhoods_10th['Nearby_Mosques'] = candidate_neighborhoods_10th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_10th['Nearby_Schools'] = candidate_neighborhoods_10th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_25th['Nearby_Mosques'] = candidate_neighborhoods_25th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_25th['Nearby_Schools'] = candidate_neighborhoods_25th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_mean['Nearby_Mosques'] = candidate_neighborhoods_mean.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_mean['Nearby_Schools'] = candidate_neighborhoods_mean.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_75th['Nearby_Mosques'] = candidate_neighborhoods_75th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_75th['Nearby_Schools'] = candidate_neighborhoods_75th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_90th['Nearby_Mosques'] = candidate_neighborhoods_90th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], mosques_df), axis=1)

    candidate_neighborhoods_90th['Nearby_Schools'] = candidate_neighborhoods_90th.apply(
        lambda row: count_nearby_schools(row['longitude'], row['latitude'], schools_df), axis=1)

    candidate_neighborhoods_10th['Combined_Score'] = (candidate_neighborhoods_10th['Distance_to_Work'] + candidate_neighborhoods_10th['Distance_to_Wife']) / 2
    candidate_neighborhoods_10th = candidate_neighborhoods_10th.sort_values(by='Combined_Score')

    candidate_neighborhoods_25th['Combined_Score'] = (candidate_neighborhoods_25th['Distance_to_Work'] + candidate_neighborhoods_25th['Distance_to_Wife']) / 2
    candidate_neighborhoods_25th = candidate_neighborhoods_25th.sort_values(by='Combined_Score')

    candidate_neighborhoods_mean['Combined_Score'] = (candidate_neighborhoods_mean['Distance_to_Work'] + candidate_neighborhoods_mean['Distance_to_Wife']) / 2
    candidate_neighborhoods_mean = candidate_neighborhoods_mean.sort_values(by='Combined_Score')

    candidate_neighborhoods_75th['Combined_Score'] = (candidate_neighborhoods_75th['Distance_to_Work'] + candidate_neighborhoods_75th['Distance_to_Wife']) / 2
    candidate_neighborhoods_75th = candidate_neighborhoods_75th.sort_values(by='Combined_Score')

    candidate_neighborhoods_90th['Combined_Score'] = (candidate_neighborhoods_90th['Distance_to_Work'] + candidate_neighborhoods_90th['Distance_to_Wife']) / 2
    candidate_neighborhoods_90th = candidate_neighborhoods_90th.sort_values(by='Combined_Score')

    recommended_neighborhoods_10th = candidate_neighborhoods_10th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_25th = candidate_neighborhoods_25th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_mean = candidate_neighborhoods_mean[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_75th = candidate_neighborhoods_75th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]
    recommended_neighborhoods_90th = candidate_neighborhoods_90th[['District', 'Distance_to_Work', 'Distance_to_Wife', 'Nearby_Schools', 'Combined_Score', 'Nearby_Mosques']]

    df_10th = recommended_neighborhoods_10th.head(2)
    df_25th = recommended_neighborhoods_25th.head(2)
    df_mean = recommended_neighborhoods_mean.head(2)
    df_75th = recommended_neighborhoods_75th.head(2)
    df_90th = recommended_neighborhoods_90th.head(2)

    df_10th['Color'] = 'Red'
    df_10th['percentile'] = '10th'
    df_25th['Color'] = 'Blue'
    df_25th['percentile'] = '25th'
    df_mean['Color'] = 'Green'
    df_mean['percentile'] = 'Mean'
    df_75th['Color'] = 'Yellow'
    df_75th['percentile'] = '75th'
    df_90th['Color'] = 'Orange'
    df_90th['percentile'] = '90th'

    combined_df = pd.concat([df_10th, df_25th, df_mean, df_75th, df_90th])

    combined_df.reset_index(drop=True, inplace=True)
    combined_df = combined_df.drop_duplicates(subset=['District'], keep='first').reset_index(drop=True)
    return combined_df

'''




def members(request):
    return render(request, 'test.html')

# views.py
# views.py
def process_form(request):
    if request.method == 'POST':
        budgets = request.POST.get('budget')
        your_location = request.POST.get('your_location')
        relative_location = request.POST.get('relative_location')

        # Process the input data (for example, parsing the locations)
        your_lat, your_lng = map(float, your_location.split(','))
        rel_lat, rel_lng = map(float, relative_location.split(','))

        # Get the recommended neighborhoods DataFrame from the pre function
        recommended_neighborhoods_df = pre(lat1=your_lat, lng1=your_lng, lat2=rel_lat, lng2=rel_lng, budget=budgets)

        # Convert the DataFrame to HTML
        results_html = recommended_neighborhoods_df.to_html(classes='table table-striped', index=False, justify='center')

        context = {
            'results_html': results_html
        }

        return render(request, 'results.html', context)
    else:
        return render(request, 'test.html')



