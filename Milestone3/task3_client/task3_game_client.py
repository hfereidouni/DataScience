import json
import requests
import task3_serving_client
from game_loader import *
from flask import Flask, jsonify, request

app = Flask(__name__)

# Internal tracker to keep track of processed events
processed_events_tracker = set()

class GameClient:
    def __init__(self):
        self.serving_client = task3_serving_client.ServingClient(port=8080)
        # self.tracked_df = pd.DataFrame()
        self.last_event = {}
        self.last_event_period = {}
        self.last_event_time_rem = {}

        #create new folder to store game data
        if not os.path.exists("event_data/"):
            os.makedirs("event_data")
            print("event folder created successfully!")
        
    def get_model_features(self):
        return self.serving_client.current_model()

    def ping_game(self, game_id):
        """
        Loads the data based on game_id into a dataframe,
        checks for new events and selects one,
        uses the selected event to return its dataframe data,
        and updates the last_event tracker with this latest event.
        """
        
        #loads data based on a game id
        game_data = load_game(game_id)

        #new game_id added
        if self.last_event.get(game_id) is None:
            temp_df = create_game_df(game_data)
            # print("in ping_game, result shape: ", temp_df.shape, temp_df.columns)
            self.last_event[game_id] = temp_df.iloc[0]['event_idx']
            self.last_event_period[game_id] = temp_df.iloc[0]['period']
            self.last_event_time_rem[game_id] = temp_df.iloc[0]['period_time_rem']

            print("tracker elements for new game_id: ", self.last_event, self.last_event_period, self.last_event_time_rem)

            #make this the new event
            new_events = int(self.last_event[game_id])
        else:
            new_events = self.check_new_event(game_data, game_id)

        result_df = pd.DataFrame()
        if new_events:
            #if any new events is_live var is True
            is_live = True            
            # sample one new event
            new_event_taken = new_events
            print("new event: ", new_event_taken, type(new_event_taken))
            
            #filter the new event
            new_events_df = create_game_df(game_data)
            new_events_df = new_events_df[new_events_df.event_idx == new_event_taken].reset_index(drop=True)
            # print("input:---", new_events_df.shape)

            # model_features = self.get_model_features()
            # predict_input_df = new_events_df[model_features]
            # predict_df = self.serving_client.predict(predict_input_df)
            # result_df = pd.merge(new_events_df, predict_df, left_index=True, right_index=True)

            # #added for xg
            # home_xg = result_df.apply(lambda row:row["chance of goal"]*1 if row["home_or_away"]=="home" else 0,axis=1)
            # away_xg = result_df.apply(lambda row:row["chance of goal"]*1 if row["home_or_away"]=="away" else 0,axis=1)
            # result_df['home_xg']=home_xg.cumsum()
            # result_df['away_xg']=away_xg.cumsum()
            result_df = new_events_df.copy()
            # print(result_df, result_df.shape, result_df.columns)
            
            #tracking purposes
            self.last_event[game_id] = result_df['event_idx'].values[0]
            self.last_event_period[game_id] = result_df['period'].values[0]
            self.last_event_time_rem[game_id] = result_df['period_time_rem'].values[0]

            period = result_df['period'].values[0]
            time_remaining = result_df['period_time_rem'].values[0]
            home_name = result_df['home_name'].values[0]
            away_name = result_df['away_name'].values[0]
            home_score = result_df['home_score'].values[0]
            away_score = result_df['away_score'].values[0]
            
            return result_df, is_live, period, time_remaining, home_name, away_name, home_score, away_score
        else:
            #No new events
            is_live=False
            return result_df, is_live, -1, -1, '', "", -1, -1 

    def time_calc(self, time_rem_str):
        minute, second = time_rem_str.split(":")
        return int(minute)*60 + int(second)
    
    def check_new_event(self, data, game_id):

        # print("New method")
        new_event = []
        for event in data['plays']:
            if event['typeDescKey'] in ['shot-on-goal', 'goal', 'missed-shot']:
                if event['period']==self.last_event_period[game_id]:
                    #compare remaining time
                    if self.time_calc(event['timeRemaining']) < self.time_calc(self.last_event_time_rem[game_id]):
                        new_event.append(event)
                elif event['period'] > self.last_event_period[game_id]:
                    # new period starts
                    new_event.append(event)
                    
        # print(f"Any new new events for {game_id}'s event {self.last_event[game_id]}: {[event['eventId']for event in new_event]}")
        #check for no new events
        if len(new_event)==0:
            return 0
        else:
            return new_event[0]["eventId"]
        return new_event


if __name__=="__main__":
    client=  GameClient()
    print("get nw event")

    game_id = "2022020011"#"2022030414"
    result = client.ping_game(game_id)
    result2 = client.ping_game(game_id)
    result2 = client.ping_game(game_id)

    # result2 = client.ping_game("2022030415")
    # result2 = client.ping_game("2022030415")
    # result2 = client.ping_game("2022030415")
    # result2 = client.ping_game("2022030415")

    # result2 = client.ping_game("2022030413")
    # result2 = client.ping_game("2022030413")

    # result2 = client.ping_game("2022030411")
    # result2 = client.ping_game("2022030411")
