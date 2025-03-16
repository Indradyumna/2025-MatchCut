import os
import cv2
import json
import argparse 
import pandas as pd
from data.netflix_matchcut.matchcut.data import Dataset
import glob

def get_df(task: str, source: str = None) -> pd.DataFrame:
    """
    Read a dataframe where `task` is one of frame or motion.
    if `source` is not provided all the data is returned,
    if it is then the data is filtered down to the pairs
    retrieved by one of the 4 heuristics.
    """
    ds = Dataset(
        task=task,
        split="train",  # ignored
        encoder_name=None,
        agg_name=None,
        source=source,
    )
    return pd.DataFrame(p.__dict__ for p in ds.pairs_labeled_all)

def get_sub_df(task: str, label: str) -> pd.DataFrame:
    df = get_df(task)
    label_bool = True if label == "pos" else False
    return df[df.label == label_bool]



def create_netflix_matchcuts_video_from_frames(data_path, variant,frame_rate):

    df = get_sub_df(*variant)
    source_data_path = data_path + f"/{variant[0]}_{variant[1]}"
    #check if the source data path exists
    assert os.path.exists(source_data_path)
    video_path = data_path + "/../match-cut_videos"
    dest_video_path = video_path + f"/{variant[0]}_{variant[1]}/fr_{frame_rate}"
    if not os.path.exists(dest_video_path):
        os.makedirs(dest_video_path)

    for idx, row in df.iterrows():
        imdb_id = row['imdb_id']
        shot1_idx = row['shot1_idx']
        shot2_idx = row['shot2_idx']
        assert shot1_idx < shot2_idx, f"shot1_idx: {shot1_idx} should be less than shot2_idx: {shot2_idx}"
        
        source_folder_path = os.path.join(source_data_path, str(imdb_id))
        dest_path = os.path.join(dest_video_path, f"{imdb_id}_{shot1_idx}_{shot2_idx}.mp4")
        
        shot1_pattern = f"shot_{str(shot1_idx).zfill(4)}_img_*.jpg"
        shot2_pattern = f"shot_{str(shot2_idx).zfill(4)}_img_*.jpg"
        images = []
        for shot_pattern in [shot1_pattern, shot2_pattern]:
            # fetch image files in sorted order using glob 
            for img_file in sorted(glob.glob(os.path.join(source_folder_path, shot_pattern))):
                print(img_file)
                img = cv2.imread(img_file)
                if img is not None:
                    images.append(img)
                else: 
                    assert False, f"Error reading {img_file}"
                    
        if images: 
            height, width, layers = images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(dest_path, fourcc, frame_rate, (width, height))
            for img in images:
                video.write(img)
        
            video.release()
            print(f'Video created for : {dest_path}')
        
    

def create_moviecuts_video_from_frames(data_path, span, frame_rate):
    video_path = data_path + "/../match-cut_videos"
    os.makedirs(video_path, exist_ok=True)
    
    video_subpath = video_path + f"/span_{span}_fr_{frame_rate}"
    os.makedirs(video_subpath, exist_ok=True)

    # Iterate through each folder in the data path
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        frames_folder = os.path.join(folder_path, 'frames')
        json_file_path = os.path.join(folder_path, f'{folder_name}.json')

        # Check if frames folder and json file exist
        if os.path.exists(frames_folder) and os.path.exists(json_file_path):
            # Load cut frame information from JSON file
            with open(json_file_path, 'r') as json_file:
                cut_info = json.load(json_file)
                cut_frame_id = cut_info.get('cut_frame_id')
                
                if cut_frame_id is not None:
                    # Collect images based on span
                    start_frame = max(0, cut_frame_id - span)
                    end_frame = cut_frame_id + span
                    images = []
                    
                    for frame_id in range(start_frame, end_frame + 1):
                        frame_file = os.path.join(frames_folder, f'{frame_id:06}.jpg')
                        if os.path.exists(frame_file):
                            img = cv2.imread(frame_file)
                            if img is not None:
                                images.append(img)
                            else: 
                                assert False, f"Image not found in location {frames_folder} and file {frame_file}"
                    if images:
                        height, width, layers = images[0].shape
                        video_folder = os.path.join(video_subpath, folder_name)
                        os.makedirs(video_folder, exist_ok=True)
                        video_file = os.path.join(video_folder, 'output_video.mp4')
                        # video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
                        video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))
                        for image in images:
                            video.write(image)
                            
                        
                        video.release()
                        print(f'Video created for folder: {video_file}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create video from frames')
    parser.add_argument('--dataset', type=str, help='Path to the data folder')
    parser.add_argument('--span', type=int, help='Number of frames to consider before and after the cut')
    parser.add_argument('--frame_rate', type=int, help='Frame rate of the output video')
    args = parser.parse_args()
    
        
    if args.dataset == "MovieCuts":
        print("Creating video from frames for MovieCuts dataset")
        data_path = "data/MovieCuts/data/framed_clips_match-cut"
        create_moviecuts_video_from_frames(data_path, args.span, args.frame_rate)
    elif args.dataset == "NetflixMC":
        data_path  = "./data/netflix_matchcut/data/movienet_netflix_filtered"
        variants = [("frame", "pos"), ("frame", "neg"), ("motion", "pos"), ("motion", "neg")]
        for variant in  variants:
            create_netflix_matchcuts_video_from_frames(data_path, variant,args.frame_rate)
    else: 
        raise ValueError("Invalid dataset name. Choose either MovieCuts or NetflixMC")
    
    

# Command to run the script
# python -m utils.create_video_from_frames --dataset MovieCuts --span 15 --frame_rate 5
# python -m utils.create_video_from_frames --dataset NetflixMC --frame_rate 5
