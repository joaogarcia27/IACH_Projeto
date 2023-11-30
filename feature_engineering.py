# -*- coding: utf-8 -*-


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pandas as pd
import pickle
import numpy as np


#----------------------------- FEATURE ENGINEERING ----------------------------


with open('2019_prem_generated_clean/2019_prem_all_stats_dict.txt', 'rb') as myFile:
    game_stats = pickle.load(myFile)


#creating a list with the team id in
team_list = []
for key in game_stats.keys():
    team_list.append(key)
team_list.sort()

#creating a dictionary with the team id as key and fixture id's as values
team_fixture_id_dict = {}
for team in team_list:
    fix_id_list = []
    for key in game_stats[team].keys():
        fix_id_list.append(key)
    sub_dict = {team:fix_id_list}
    team_fixture_id_dict.update(sub_dict)

#the list of fixtures was first home then away, we want them in chronological order so we need to sort them.
for team in team_fixture_id_dict:
    team_fixture_id_dict[team].sort()


#creating a running average function
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x.astype(float), 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
 

#we can now iterate over the fixture ID list given a team id key using the dict created above. N.B./ the number of games over which the past data is averaged. A large number will smooth out past performance where as a small number will result in the prediction being heavily reliant on very recent form. This is worth testing the ml model build phase

def generate_ml_df(games_slide):
  
    #creating final features which will be appended
    t_total_shots = []
    t_shots_inside_box = []     
    t_fouls = []
    t_corners = []
    t_posession = []
    t_pass_accuracy = []
    t_goals = []
    t_goals_target = []
    o_total_shots = []
    o_shots_inside_box = []     
    o_fouls = []
    o_corners = []
    o_posession = []
    o_pass_accuracy = []
    o_goals = []
    o_goals_target = []
    fix_id = []
    result_indicator = []
    total_corners = []
    
    for team_id in team_list[:]:
        team = game_stats[team_id] #team dictionary
        
        #creating the initial features - it is important these get overwritten with each iteration
        team_total_shots = []
        team_shots_inside_box = []     
        team_fouls = []
        team_corners = []
        team_posession = []
        team_pass_accuracy = []
        team_goals = []
        opponent_total_shots = []
        opponent_shots_inside_box = []     
        opponent_fouls = []
        opponent_corners = []
        opponent_posession = []
        opponent_pass_accuracy = []
        opponent_goals = []
        result_indicator_raw = []
        total_corners_raw = []
        
        
        #iterating over the fixture id to create feature lists
        for game_id in team_fixture_id_dict[team_id]:
            game = team[game_id] #game df
            temp_index = pd.Index(game['Team Identifier'])
            team_ind = temp_index.get_loc(1)
            opponent_ind = temp_index.get_loc(2)
            
            #team and opponent pseudo features: list of raw feature data for each game
            team_total_shots.append(game['Total Shots'].iloc[team_ind])

            team_shots_inside_box.append(game['Shots insidebox'].iloc[team_ind])
            team_fouls.append(game['Fouls'].iloc[team_ind])
            team_corners.append(game['Corner Kicks'].iloc[team_ind])
            team_posession.append(game['Ball Possession'].str.replace('[\%]', '').iloc[team_ind])
            team_pass_accuracy.append(game['Passes %'].iloc[team_ind])
            team_goals.append(game['Goals'].iloc[team_ind])
            opponent_total_shots.append(game['Total Shots'].iloc[opponent_ind])
            opponent_shots_inside_box.append(game['Shots insidebox'].iloc[opponent_ind])
            opponent_fouls.append(game['Fouls'].iloc[opponent_ind])
            opponent_corners.append(game['Corner Kicks'].iloc[opponent_ind])
            opponent_posession.append(game['Ball Possession'].str.replace('[\%]', '').iloc[opponent_ind])
            opponent_pass_accuracy.append(game['Passes %'].iloc[opponent_ind])
            opponent_goals.append(game['Goals'].iloc[opponent_ind])
            result_indicator_raw.append(game['Points'].iloc[team_ind])
            total_corners_raw.append(game['Corner Kicks'].iloc[team_ind]+game['Corner Kicks'].iloc[opponent_ind])


            #sliding average of the raw feature lists above to create the final features
        team_total_shots_slide = running_mean(np.array(team_total_shots), games_slide)[:-1]
        team_shots_inside_box_slide = running_mean(np.array(team_shots_inside_box), games_slide)[:-1]
        team_fouls_slide = running_mean(np.array(team_fouls), games_slide)[:-1]
        team_corners_slide = running_mean(np.array(team_corners), games_slide)[:-1]
        team_posession_slide = running_mean(np.array([float(possession.replace('%', '')) for possession in team_posession]), games_slide)[:-1]
        team_pass_accuracy_slide = running_mean(np.array([float(accuracy.replace('%', '')) for accuracy in team_pass_accuracy]), games_slide)[:-1]
        team_goals_slide = running_mean(np.array(team_goals), games_slide)[:-1]
        team_goals_target = team_goals[games_slide:]
        opponent_total_shots_slide = running_mean(np.array(opponent_total_shots), games_slide)[:-1]
        opponent_shots_inside_box_slide = running_mean(np.array( opponent_shots_inside_box), games_slide)[:-1]
        opponent_fouls_slide = running_mean(np.array(opponent_fouls), games_slide)[:-1]
        opponent_corners_slide = running_mean(np.array(opponent_corners), games_slide)[:-1]
        opponent_posession_slide = running_mean(np.array([float(possession.replace('%', '')) for possession in opponent_posession]), games_slide)[:-1]
        opponent_pass_accuracy_slide = running_mean(np.array([float(accuracy.replace('%', '')) for accuracy in opponent_pass_accuracy]), games_slide)[:-1]
        opponent_goals_slide = running_mean(np.array(opponent_goals), games_slide)[:-1]
        opponent_goals_target = opponent_goals[games_slide:]
        fix_id_slide = team_fixture_id_dict[team_id][games_slide:]
        result_indicator_slide = result_indicator_raw[games_slide:]
        total_corners_slide = total_corners_raw[games_slide:]

    
        #appending over the iterables, the above variables will be overwritten with each iteration
        t_total_shots.extend(team_total_shots_slide)
        t_shots_inside_box.extend(team_shots_inside_box_slide)
        t_fouls.extend(team_fouls_slide)
        t_corners.extend(team_corners_slide)
        t_posession.extend(team_posession_slide)
        t_pass_accuracy.extend(team_pass_accuracy_slide)
        t_goals.extend(team_goals_slide)
        t_goals_target.extend(team_goals_target)
        o_total_shots.extend(opponent_total_shots_slide)
        o_shots_inside_box.extend(opponent_shots_inside_box_slide)
        o_fouls.extend(opponent_fouls_slide)
        o_corners.extend(opponent_corners_slide)
        o_posession.extend(opponent_posession_slide)
        o_pass_accuracy.extend(opponent_pass_accuracy_slide)
        o_goals.extend(opponent_goals_slide)
        o_goals_target.extend(opponent_goals_target)
        fix_id.extend(fix_id_slide)
        result_indicator.extend(result_indicator_slide)
        total_corners.extend(total_corners_slide)


    
    #piecing together the results into a dataframe   
    df_ready_for_ml = pd.DataFrame({})  
    df_ready_for_ml['Team Av Shots'] = t_total_shots
    df_ready_for_ml['Team Av Shots Inside Box'] = t_shots_inside_box
    df_ready_for_ml['Team Av Fouls'] = t_fouls
    df_ready_for_ml['Team Av Corners'] = t_corners
    df_ready_for_ml['Team Av Possession'] = t_posession
    df_ready_for_ml['Team Av Pass Accuracy'] = t_pass_accuracy
    df_ready_for_ml['Team Av Goals'] = t_goals
    df_ready_for_ml['Opponent Av Shots'] = o_total_shots
    df_ready_for_ml['Opponent Av Shots Inside Box'] = o_shots_inside_box
    df_ready_for_ml['Opponent Av Fouls'] = o_fouls
    df_ready_for_ml['Opponent Av Corners'] = o_corners
    df_ready_for_ml['Opponent Av Possession'] = o_posession
    df_ready_for_ml['Opponent Av Goals'] = o_goals
    df_ready_for_ml['Opponent Av Pass Accuracy'] = o_pass_accuracy
    df_ready_for_ml['Team Goal Target'] = t_goals_target
    df_ready_for_ml['Opponent Goal Target'] = o_goals_target
    df_ready_for_ml['Target Fixture ID'] = fix_id
    df_ready_for_ml['Result Indicator'] = result_indicator
    df_ready_for_ml['Total Corners'] = total_corners
    print(df_ready_for_ml['Total Corners'])
    
    #returning the dataframe
    return df_ready_for_ml


#creating and saving the ml dataframe with a 5 game sliding average.
df_ready_for_ml_5 = generate_ml_df(5)
with open('2019_prem_generated_clean/2019_prem_df_for_ml_5.txt', 'wb') as myFile:
    pickle.dump(df_ready_for_ml_5, myFile)

#creating and saving the ml dataframe with a 10 game sliding average.
df_ready_for_ml_10 = generate_ml_df(10)
with open('2019_prem_generated_clean/2019_prem_df_for_ml_10.txt', 'wb') as myFile:
    pickle.dump(df_ready_for_ml_10, myFile)


# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')