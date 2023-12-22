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
        self.tracked_df = pd.DataFrame()
        self.last_event = {}
        self.last_event_period = {}
        self.last_event_time_rem = {}
        if not os.path.exists("event_data/"):
            os.makedirs("event_data")
            print("event folder created successfully!")
        
    def know_current_model(self):
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
        new_events = self.check_new_event(game_data, game_id)

        result_df = pd.DataFrame()
        if len(new_events) > 0:
            # sample one new event
            new_event_taken = new_events[0]["eventId"]
            
            #if any new events is_live var is True
            is_live = True
            
            new_events_df = create_game_df(game_data)
            #filter the new event
            new_events_df = new_events_df[new_events_df.event_idx == new_event_taken]
            model_features = self.know_current_model()
            
            # predict_input_df = new_events_df[model_features]
            result_df = new_events_df.copy()

            self.tracked_df = pd.concat([self.tracked_df, result_df],ignore_index=True)

            #added for xg
            predict_df = self.serving_client.predict(self.tracked_df[model_features])
            self.tracked_df= pd.concat([self.tracked_df,predict_df],axis=1)
            home_xg = self.tracked_df.apply(lambda row:row["chance of goal"]*1 if row["home_or_away"]=="home" else 0,axis=1)
            away_xg = self.tracked_df.apply(lambda row:row["chance of goal"]*1 if row["home_or_away"]=="away" else 0,axis=1)
            self.tracked_df['home_xg']=home_xg.cumsum()
            self.tracked_df['away_xg']=away_xg.cumsum()
            print(self.tracked_df)

            self.last_event[game_id] = result_df['event_idx'].values[0]
            self.last_event_period[game_id] = result_df['period'].values[0]
            self.last_event_time_rem[game_id] = result_df['period_time_rem'].values[0]
            print("tracker elements in ping step: ", self.last_event, self.last_event_period, self.last_event_time_rem)
            self.tracked_df.to_csv(f"event_data/game_{game_id}.csv")

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

    def get_game(self, game_id):
        """
        start of pinging process,
        loads data from local if game_id data already present, else create it from response,
        adds first event in tracker,
        and finally runs ping_game() function.
        """
        # adds data to local vent folder for a new game_id
        if self.last_event.get(game_id) is None: 
            game_df = create_game_df(load_game(game_id))
            # print("game df shape: ", game_df.shape)
            model_features = self.know_current_model()
            predict_input_df = game_df[model_features] #only for shot_dist_only model
            # predict_df = self.serving_client.predict(predict_input_df)
            # result_df = pd.merge(game_df, predict_df, left_index=True, right_index=True)
            result_df = game_df.copy()
            print("Newly joined \n")
            # print("result shape: ", result_df.shape, result_df.columns)
            self.tracked_df = result_df
            self.last_event[game_id] = result_df.iloc[0]['event_idx']
            self.last_event_period[game_id] = result_df.iloc[0]['period']
            self.last_event_time_rem[game_id] = result_df.iloc[0]['period_time_rem']

            print("tracker elements: ", self.last_event, self.last_event_period, self.last_event_time_rem)
            self.tracked_df.to_csv(f"event_data/game_{game_id}.csv")
        else:
            print("Alreay have it!")
            self.tracked_df = pd.read_csv(f'event_data/game_{game_id}.csv')
        
        self.ping_game(game_id)

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
                    
        # print("Prev method")            
        # new_event = [event for event in data['plays'] if event['typeDescKey'] in ['shot-on-goal', 'goal', 'missed-shot'] and int(event['eventId']) > self.last_event[game_id]]
        
        # print(f"Any new new events for {game_id}'s event {self.last_event[game_id]}: {[event['eventId']for event in new_event]}")
        return new_event


if __name__=="__main__":
    client=  GameClient()
    print("get nw event")

    game_id = "2022030414"
    result = client.get_game(game_id)
    # result2 = client.get_game(game_id)
    # result2 = client.get_game(game_id)

    # result2 = client.get_game("2022030415")
    # result2 = client.get_game("2022030415")
    # result2 = client.get_game("2022030415")
    # result2 = client.get_game("2022030415")

    # result2 = client.get_game("2022030413")
    # result2 = client.get_game("2022030413")

    # result2 = client.get_game("2022030411")
    # result2 = client.get_game("2022030411")
    # print("get result 2 shape:", type(result2))

    # result2 = client.get_game("2022030415")
    # print("get result 2 shape:", result2)
    
    # result2 = client.ping_game("2022030415")
    # print("ping result 2 shape:", result2)

# @app.route("/get_live_events", methods=["GET"])
# def get_live_events():
#     """pd.DataFrame.append
#     Simulates querying live game data and processing events.
#     Filters out events that have already been processed.
#     """
#     try:
#         # Simulate querying live game data from the NHL API
#         game_id = "2022030411"
#         # live_events_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/feed/live"
#         live_events_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
#         response = requests.get(live_events_url)

#         if response.status_code == 200:
#             live_events_data = response.json()

#             # Filter out events that have already been processed
#             new_events = filter_unprocessed_events(live_events_data["events"])

#             # Process the new events (e.g., produce features, query prediction service)
#             processed_results = process_events(new_events)

#             # Update the processed events tracker
#             update_processed_events_tracker(new_events)

#             return jsonify(processed_results)
#         else:
#             return jsonify({"error": f"Failed to fetch live events. Status code: {response.status_code}"}), 500

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# def filter_unprocessed_events(events):
#     """Filter out events that have already been processed."""
#     return [event for event in events if event["event_id"] not in processed_events_tracker]

# def process_events(events):
#     """Simulate processing events (e.g., producing features, querying prediction service)."""
#     processed_results = []

#     for event in events:
#         # Simulate processing each event
#         # (Replace this with your actual logic for processing events)
#         processed_result = {
#             "event_id": event["event_id"],
#             "event_type": event["event_type"],
#             "processed_data": "Some processed data",
#         }
#         processed_results.append(processed_result)

#     return processed_results

# def update_processed_events_tracker(new_events):
#     """Update the processed events tracker with the IDs of newly processed events."""
#     processed_events_tracker.update(event["event_id"] for event in new_events)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8081)
