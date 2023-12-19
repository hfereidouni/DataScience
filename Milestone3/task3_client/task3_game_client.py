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
        self.serving_client = task3_serving_client.ServingClient()
        self.tracked_df = pd.DataFrame()
        self.last_event = {}

    def ping_game(self, game_id):
        game_data = load_game(game_id)
        new_event = self.check_new_event(game_data)

        result_df = pd.DataFrame()
        if len(new_events) > 0:
            new_events_df = create_game_df(game_data)
            predict_df = self.serving_client.ppredict(new_events_df)
            result_df = pd.merge(new_events_df, predict_df, left_index=True, right_index=True)
            self.tracked_df = pd.concat([self.tracked_df, result_df])
            self.last_event[game_id] = result_df.iloc[-1]['event_id']
            self.tracker.to_csv(f"game_{game_id}.csv")

        return result_df

    def get_game(self, game_id):
        if self.last_event.get(game_id) is None:
            game_df = create_game_df(game_data)
            predict_df = self.serving_client.predict(game_df)
            result_df = pd.merge(game_df, predict_df, left_index=True, right_index=True)
            self.tracked_df = result_df
            self.last_event[game_id] = result_df.iloc[-1]['event_id']
            self.tracker.to_csv(f"game_{game_id}.csv")
        else:
            self.tracker = pd.read_csv(f'game_{game_id}.csv')
            self.ping_game(game_id)

    def check_new_event(self, data):
        new_event = [event for event in data['plays'] if int(event['eventId']) > self.last_event_id[game_id] and event['typeDescKey'] in ['shot-on-goal', 'goal']]
        return new_event


if __name__=="__main__":
    client=  GameClient()
    print("get nw event")

    client.get_game()

# @app.route("/get_live_events", methods=["GET"])
# def get_live_events():
#     """
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
