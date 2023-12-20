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
        if not os.path.exists("event_data/"):
            os.makedirs("event_data")
            print("event folder created successfully!")
        
    def know_current_model(self):
        return self.serving_client.current_model()

    def ping_game(self, game_id):
        #loads data based on a game id
        game_data = load_game(game_id)
        new_events = self.check_new_event(game_data, game_id)

        result_df = pd.DataFrame()
        if len(new_events) > 0:
            
            #if any new events is_live var is True
            is_live = True
            
            new_events_df = create_game_df(game_data)
            print(f"events new: {len(new_events_df.event_idx.unique())}")
            model_features = self.know_current_model()
            
            predict_input_df = new_events_df[model_features]
            # print(f"data shape: {new_events_df.shape} predict shape: {predict_input_df.shape}")
            predict_df = self.serving_client.predict(predict_input_df)
            # print(f"predict shape: {predict_df.shape} {predict_df.columns} {predict_df.index}")
            # print(f"new event df: {new_events_df.index}")
            # print(predict_df)
            result_df = pd.merge(new_events_df, predict_df, left_index=True, right_index=True)
            print(result_df.columns)
            self.tracked_df = pd.concat([self.tracked_df, result_df])
            self.last_event[game_id] = result_df.iloc[-1]['event_idx']
            self.tracked_df.to_csv(f"event_data/game_{game_id}.csv")

            period = result_df['period'].values
            time_remaining = result_df['period_time_rem'].values
            home_name = result_df['home_name'].values
            away_name = result_df['away_name'].values
            home_score = result_df['home_score'].values
            away_score = result_df['away_score'].values
            
            return result_df, is_live, period, time_remaining, home_name, away_name, home_score, away_score
        else:
            is_live=False
            return result_df, is_live, -1, -1, '', "", -1, -1 

    def get_game(self, game_id):
        # adds data to local vent folder for a new game_id
        if self.last_event.get(game_id) is None: 
            game_df = create_game_df(load_game(game_id))
            # print("game df shape: ", game_df.shape)
            model_features = self.know_current_model()
            predict_input_df = game_df[model_features] #only for shot_dist_only model
            predict_df = self.serving_client.predict(predict_input_df)
            # print("predict shape: ", predict_df.shape, predict_df.columns, predict_df.index)
            # print(predict_df)
            # print("data shape:", game_df.index)
            result_df = pd.merge(game_df, predict_df, left_index=True, right_index=True)
            # result_df = game_df.copy()
            print("result shape: ", result_df.shape, result_df.columns)
            self.tracked_df = result_df
            self.last_event[game_id] = result_df.iloc[-1]['event_idx']
            print("tracker elements: ", self.last_event)
            self.tracked_df.to_csv(f"event_data/game_{game_id}.csv")
        else:
            self.tracked_df = pd.read_csv(f'event_data/game_{game_id}.csv')
            self.ping_game(game_id)

    def check_new_event(self, data, game_id):

        new_event = [event for event in data['plays'] if int(event['eventId']) > self.last_event[game_id] and event['typeDescKey'] in ['shot-on-goal', 'goal']]
        print(f"Any new new events: {new_event}")
        return new_event


if __name__=="__main__":
    client=  GameClient()
    print("get nw event")

    game_id = "2022030411"
    result = client.get_game(game_id)
    # print("resultt shape: ", type(result))

    result2 = client.ping_game(game_id)
    print("ping result 2 shape:", result2)

    # result2 = client.get_game("2022030414")
    # print("get result 2 shape:", type(result2))

    result2 = client.get_game("2022030415")
    print("get result 2 shape:", result2)
    
    result2 = client.ping_game("2022030415")
    print("ping result 2 shape:", result2)

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
