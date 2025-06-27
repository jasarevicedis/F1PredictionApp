import fastf1

schedule = fastf1.get_event_schedule(2025)

track_names = schedule['Location'].tolist()

for track in track_names:
    print(track)