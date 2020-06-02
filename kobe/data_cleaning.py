import pandas as pd
data = pd.read_csv('/cos_person/kobe_predict/kobe.csv')

data.set_index('shot_id', inplace=True)
data["action_type"] = data["action_type"].astype('object')
data["combined_shot_type"] = data["combined_shot_type"].astype('category')
data["game_event_id"] = data["game_event_id"].astype('category')
data["game_id"] = data["game_id"].astype('category')
data["period"] = data["period"].astype('object')
data["playoffs"] = data["playoffs"].astype('category')
data["season"] = data["season"].astype('category')
data["shot_made_flag"] = data["shot_made_flag"].astype('category')
data["shot_type"] = data["shot_type"].astype('category')
data["team_id"] = data["team_id"].astype('category')

data.drop('team_id', axis=1, inplace=True) # Always one number
data.drop('lat', axis=1, inplace=True) # Correlated with loc_x
data.drop('lon', axis=1, inplace=True) # Correlated with loc_y
data.drop('game_id', axis=1, inplace=True) # Independent
data.drop('game_event_id', axis=1, inplace=True) # Independent
data.drop('team_name', axis=1, inplace=True) # Always LA Lakers
data.drop('shot_made_flag', axis=1, inplace=True)

data.to_csv('/cos_person/kobe_predict/kobe_cleanning.csv', index=False)