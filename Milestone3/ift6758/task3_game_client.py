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
        self.serving_client = task3_serving_client.ServingClient(port=5000)
        # self.tracked_df = pd.DataFrame()
        self.last_event = {}
        self.last_event_period = {}
        self.last_event_time_rem = {}
        self.first_fetch_done = 0

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
        print(type(game_data))

        #check for previous game
        if game_data["plays"][-1]["typeDescKey"]=="game-end":
            print("Previous game loaded")
            temp_df = create_game_df(game_data)
            all_new_events = self.collect_all_new_events(game_data, game_id)
            temp_df = temp_df[temp_df.event_idx.isin(all_new_events)].reset_index(drop=True)
            period = temp_df.iloc[-1]['period'].astype(int)
            time_remaining = temp_df.iloc[-1]['period_time_rem']
            home_name = temp_df.iloc[-1]['home_name']
            away_name = temp_df.iloc[-1]['away_name']
            home_score = temp_df.iloc[-1]['home_score'].astype(int)
            away_score = temp_df.iloc[-1]['away_score'].astype(int)
            
            return temp_df, False, period, time_remaining, home_name, away_name, home_score, away_score

        #new game_id added
        # print("check for first fetch: ", self.first_fetch_done)
        if self.last_event.get(game_id) is None and self.first_fetch_done==0:
            temp_df = create_game_df(game_data)
            #collect all shot data uptil now
            all_new_events = self.collect_all_new_events(game_data, game_id)
            # print(all_new_events)
            if len(all_new_events)==0:
                print("No new shot events during first fetch")
                return pd.DataFrame(), True, -1, -1, '', "", -1, -1
            else:
                temp_df = temp_df[temp_df.event_idx.isin(all_new_events)].reset_index(drop=True)
                # print("all new evetns df: ", temp_df.shape)
                #saving lastest shot event
                self.last_event[game_id] = temp_df.iloc[-1]['event_idx']
                self.last_event_period[game_id] = temp_df.iloc[-1]['period']
                self.last_event_time_rem[game_id] = temp_df.iloc[-1]['period_time_rem']
                print("tracker elements for new game_id: ", self.last_event, self.last_event_period, self.last_event_time_rem)
                print("New live game with first shot events")
                self.first_fetch_done=1

                period = temp_df.iloc[-1]['period'].astype(int)
                time_remaining = temp_df.iloc[-1]['period_time_rem']
                home_name = temp_df.iloc[-1]['home_name']
                away_name = temp_df.iloc[-1]['away_name']
                home_score = temp_df.iloc[-1]['home_score'].astype(int)
                away_score = temp_df.iloc[-1]['away_score'].astype(int)
                print(period, time_remaining, home_name, away_name, home_score, away_score)
                return temp_df, True, period, time_remaining, home_name, away_name, home_score, away_score
        else:
            new_events = self.check_new_event(game_data, game_id)
            # print("enter me new events latest: ", new_events)
            result_df = pd.DataFrame()
            if new_events:
                #if any new events is_game_live var is True
                is_game_live = True            
                # sample one new event
                new_event_taken = new_events
                # print("new event: ", new_event_taken, type(new_event_taken))
                
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
                
                return result_df, is_game_live, period, time_remaining, home_name, away_name, home_score, away_score
            else:
                #No new events
                is_game_live=True
                return result_df, is_game_live, self.last_event_period[game_id], self.last_event_time_rem[game_id], '', "", -1, -1 

    def time_calc(self, time_rem_str):
        minute, second = time_rem_str.split(":")
        return int(minute)*60 + int(second)

    def collect_all_new_events(self, data, game_id):
        
        new_event = []
        for event in data['plays']:
            if event['typeDescKey'] in ['shot-on-goal', 'goal', 'missed-shot']:
                new_event.append(event["eventId"])

        return new_event

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

    game_id = "2023020513" #"2022030414" #"2023020510" #  #
    result = client.ping_game(game_id)
    print("==========================================")
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
