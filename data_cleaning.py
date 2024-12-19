import pandas as pd
import os
import warnings
import numpy as np

# Suppress SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def filter_players_by_position(raw_data_path, valid_positions):
    """Filter players by position and return valid NFL IDs"""
    print("Filtering players by position...")
    players_df = pd.read_csv(os.path.join(raw_data_path, 'players.csv'))
    filtered_players_df = players_df.loc[players_df['position'].isin(valid_positions)].copy()
    filtered_players_df['nflId'] = pd.to_numeric(filtered_players_df['nflId'], errors='coerce').dropna().astype(int)
    print("Filtered players by position. Total players found:", len(filtered_players_df))
    return filtered_players_df['nflId'].unique()

def filter_tracking_data(raw_data_path, valid_nfl_ids, weeks=range(1, 10), chunk_size=50000):
    """Filter tracking data for offensive players only and 'BEFORE_SNAP'/'SNAP' frameType"""
    print("Filtering tracking data...")
    all_weeks_data = []
    for week in weeks:
        print(f"Processing tracking data for week {week}...")
        tracking_file_path = os.path.join(raw_data_path, f'tracking_week_{week}.csv')
        week_data = []
        for chunk in pd.read_csv(tracking_file_path, chunksize=chunk_size):
            filtered_chunk = chunk.loc[chunk['nflId'].isin(valid_nfl_ids) & chunk['frameType'].isin(['BEFORE_SNAP', 'SNAP'])].copy()
            filtered_chunk['gameId'] = pd.to_numeric(filtered_chunk['gameId'], errors='coerce').dropna().astype(int)
            filtered_chunk['playId'] = pd.to_numeric(filtered_chunk['playId'], errors='coerce').dropna().astype(int)
            filtered_chunk['nflId'] = pd.to_numeric(filtered_chunk['nflId'], errors='coerce').dropna().astype(int)
            if not filtered_chunk.empty:
                filtered_chunk.loc[:, 'week'] = week
                week_data.append(filtered_chunk)
        if week_data:
            all_weeks_data.append(pd.concat(week_data, ignore_index=True))
    print("Tracking data filtered and combined.")
    return pd.concat(all_weeks_data, ignore_index=True) if all_weeks_data else pd.DataFrame()

def filter_for_pass_plays(raw_data_path, tracking_data):
    """Filter for pass plays only (isDropback = TRUE)"""
    print("Filtering for pass plays...")
    plays_df = pd.read_csv(os.path.join(raw_data_path, 'plays.csv'))
    plays_df['isDropback'] = plays_df['isDropback'].replace({True: 'TRUE', False: 'FALSE'}).fillna('FALSE').astype(str).str.upper()
    dropback_play_ids = plays_df.loc[plays_df['isDropback'] == 'TRUE', ['gameId', 'playId']].copy()
    dropback_play_ids['gameId'] = pd.to_numeric(dropback_play_ids['gameId'], errors='coerce').dropna().astype(int)
    dropback_play_ids['playId'] = pd.to_numeric(dropback_play_ids['playId'], errors='coerce').dropna().astype(int)
    filtered_data = tracking_data.merge(dropback_play_ids, on=['gameId', 'playId'], how='inner')
    print("Filtered pass plays. Total rows after filtering:", len(filtered_data))
    return filtered_data

def merge_with_plays(raw_data_path, tracking_data):
    """Merge tracking data with relevant columns from plays.csv"""
    print("Merging tracking data with plays data...")
    relevant_columns = ['gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'yardlineSide', 'yardlineNumber', 'gameClock', 'absoluteYardlineNumber']
    plays_df = pd.read_csv(os.path.join(raw_data_path, 'plays.csv'))[relevant_columns].copy()
    plays_df['gameId'] = pd.to_numeric(plays_df['gameId'], errors='coerce').dropna().astype(int)
    plays_df['playId'] = pd.to_numeric(plays_df['playId'], errors='coerce').dropna().astype(int)
    merged_data = tracking_data.merge(plays_df, on=['gameId', 'playId'], how='left')
    print("Tracking data merged with plays data. Total rows after merge:", len(merged_data))
    return merged_data

def filter_player_play_data(raw_data_path):
    """Filter player play data where routeRan is not null and keep specific columns"""
    print("Filtering player play data...")
    player_play_df = pd.read_csv(os.path.join(raw_data_path, 'player_play.csv'))
    player_play_df['gameId'] = pd.to_numeric(player_play_df['gameId'], errors='coerce').dropna().astype(int)
    player_play_df['playId'] = pd.to_numeric(player_play_df['playId'], errors='coerce').dropna().astype(int)
    player_play_df['nflId'] = pd.to_numeric(player_play_df['nflId'], errors='coerce').dropna().astype(int)
    filtered_player_play_df = player_play_df.loc[player_play_df['routeRan'].notnull(), ['gameId', 'playId', 'nflId', 'routeRan']].copy()
    print("Filtered player play data. Total rows after filtering:", len(filtered_player_play_df))
    return filtered_player_play_df

def convert_game_clock_to_seconds(game_clock):
    """Convert game clock from MM:SS format to seconds."""
    try:
        minutes, seconds = map(int, game_clock.split(':'))
        return minutes * 60 + seconds
    except Exception as e:
        return np.nan

def calculate_player_distances(df):
    distances = []
    for (gameId, playId, frameId), frame_group in df.groupby(['gameId', 'playId', 'frameId']):
        frame_players = frame_group[['nflId', 'x', 'y']].values
        for i in range(len(frame_players)):
            distances_for_player = []
            for j in range(len(frame_players)):
                if i != j:
                    x1, y1 = frame_players[i][1], frame_players[i][2]
                    x2, y2 = frame_players[j][1], frame_players[j][2]
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    distances_for_player.append(distance)
            distances.append({
                'gameId': gameId,
                'playId': playId,
                'frameId': frameId,
                'nflId': frame_players[i][0],
                'min_distance': np.min(distances_for_player),
                'max_distance': np.max(distances_for_player),
                'mean_distance': np.mean(distances_for_player),
                'std_distance': np.std(distances_for_player)
            })
    return pd.DataFrame(distances)

def main():
    data_dir = os.path.join(os.getcwd(), 'data')
    raw_data_path = os.path.join(data_dir, 'raw')
    combined_path = os.path.join(data_dir, 'combined')
    os.makedirs(combined_path, exist_ok=True)
    
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'T', 'G', 'C']
    
    print("Starting player position filtering...")
    valid_nfl_ids = filter_players_by_position(raw_data_path, valid_positions)
    
    print("Starting tracking data filtering...")
    tracking_data = filter_tracking_data(raw_data_path, valid_nfl_ids)
    
    print("Filtering for pass plays...")
    pass_play_data = filter_for_pass_plays(raw_data_path, tracking_data)
    
    print("Merging tracking data with plays data...")
    final_tracking_data = merge_with_plays(raw_data_path, pass_play_data)
    
    print("Converting game clock to seconds...")
    final_tracking_data['gameClockInSeconds'] = final_tracking_data['gameClock'].apply(convert_game_clock_to_seconds)
    
    print("Calculating player distances...")
    distance_metrics = calculate_player_distances(final_tracking_data)
    
    final_tracking_data = final_tracking_data.merge(distance_metrics, on=['gameId', 'playId', 'frameId', 'nflId'], how='left')
    
    tracking_columns = ['gameId', 'playId', 'nflId', 'displayName', 'frameId', 'frameType', 'time', 'jerseyNumber', 'club', 
                        'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'event', 'week', 'quarter', 'down', 
                        'yardsToGo', 'yardlineSide', 'yardlineNumber', 'gameClock', 'absoluteYardlineNumber', 
                        'gameClockInSeconds', 'min_distance', 'max_distance', 'mean_distance', 'std_distance']
    available_columns = [col for col in tracking_columns if col in final_tracking_data.columns]
    final_tracking_data = final_tracking_data[available_columns]
    
    # Save the final tracking data to CSV
    tracking_output_path = os.path.join(combined_path, 'final_tracking_data.csv')
    final_tracking_data.to_csv(tracking_output_path, index=False)
    print(f"Final tracking data saved to {tracking_output_path}")
    
    # Filter player play data
    player_play_data = filter_player_play_data(raw_data_path)
    
    # Save the final player play data to CSV
    player_play_output_path = os.path.join(combined_path, 'final_player_play_data.csv')
    player_play_data.to_csv(player_play_output_path, index=False)
    print(f"Final player play data saved to {player_play_output_path}")
    
    print("Data processing complete.")
    return final_tracking_data, player_play_data

if __name__ == "__main__":
    final_tracking_data, player_play_data = main()