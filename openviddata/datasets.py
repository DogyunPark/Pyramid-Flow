import csv
import os
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import cv2

from torchvision.transforms.functional import InterpolationMode
from . import video_transforms
from .utils import center_crop_arr

import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import ipdb

import argparse, os, sys, glob
import datetime, time
#from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

from PIL import Image
import random


def get_transform(width, height, new_width=None, new_height=None, resize=True):
    transform_list = []
    transform_list.append(video_transforms.RandomHorizontalFlipVideo())
    if resize:
        # rescale according to the largest ratio
        scale = max(new_width / width, new_height / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        
        transform_list.append(transforms.Resize((resized_height, resized_width), InterpolationMode.BICUBIC, antialias=True))
        transform_list.append(transforms.CenterCrop((new_height, new_width)))
    
    transform_list.extend([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_list = transforms.Compose(transform_list)

    return transform_list


def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


class VideoFrameProcessor:
    # load a video and transform
    def __init__(self, transform_list, num_frames=24, sample_fps=24):
        print(f"Transform List is {transform_list}")
        self.num_frames = num_frames
        self.transform = transforms.Compose(transform_list)
        self.sample_fps = sample_fps

    def __call__(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frames = []

            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)

            video_capture.release()
            sample_fps = self.sample_fps
            interval = max(int(fps / sample_fps), 1)
            frames = frames[::interval]

            if len(frames) < self.num_frames:
                num_frame_to_pack = self.num_frames - len(frames)
                recurrent_num = num_frame_to_pack // len(frames)
                frames = frames + recurrent_num * frames + frames[:(num_frame_to_pack % len(frames))]
                assert len(frames) >= self.num_frames, f'{len(frames)}'

            start_indexs = list(range(0, max(0, len(frames) - self.num_frames + 1)))            
            start_index = random.choice(start_indexs)

            filtered_frames = frames[start_index : start_index+self.num_frames]
            assert len(filtered_frames) == self.num_frames, f"The sampled frames should equals to {self.num_frames}"

            filtered_frames = torch.stack(filtered_frames).float() / 255
            filtered_frames = self.transform(filtered_frames)
            filtered_frames = filtered_frames.permute(1, 0, 2, 3)

            return filtered_frames, None
            
        except Exception as e:
            print(f"Load video: {video_path} Error, Exception {e}")
            return None, None

class DatasetFromCSVAndJSON(torch.utils.data.Dataset):
    """Load video according to both csv and json files.

    Args:
        csv_path (str): Path to the CSV file.
        json_path (str): Path to the JSON file.
        num_frames (int): Number of video frames to load.
        frame_interval (int): Interval between frames.
        transform (callable): Transform to apply to video frames.
        csv_root (str): Root directory containing video files from CSV.
        json_root (str): Root directory containing video files from JSON.
    """

    def __init__(
        self,
        csv_path,
        json_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        csv_root=None,
        json_root=None
    ):
        video_samples = []

        # Load from CSV
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            csv_list = list(reader)
        for vid in csv_list[1:]:  # no csv head
            vid_name = vid[0]
            vid_path = os.path.join(csv_root, vid_name)
            vid_caption = vid[1]
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])

        # Load from JSON
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for entry in json_data:
            vid_name = entry['vid']
            vid_path = os.path.join(json_root, f"{vid_name}.mp4")
            vid_caption = entry['caption']
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])

        self.samples = video_samples
        self.is_video = True
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]

        if self.is_video:
            is_exit = os.path.exists(path)
            if is_exit:
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                total_frames = len(vframes)
            else:
                total_frames = 0
            
            loop_index = index
            while(total_frames < self.num_frames or is_exit == False):
                loop_index += 1
                if loop_index >= len(self.samples):
                    loop_index = 0
                sample = self.samples[loop_index]
                path = sample[0]
                text = sample[1]

                is_exit = os.path.exists(path)
                if is_exit:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                    total_frames = len(vframes)
                else:
                    total_frames = 0
            #  video exits and total_frames >= self.num_frames
            
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
            
            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)

class DatasetFromCSVAndJSON2(torch.utils.data.Dataset):
    """Load video according to both csv and json files.

    Args:
        csv_path (str): Path to the CSV file.
        json_path (str): Path to the JSON file.
        num_frames (int): Number of video frames to load.
        frame_interval (int): Interval between frames.
        transform (callable): Transform to apply to video frames.
        csv_root (str): Root directory containing video files from CSV.
        json_root (str): Root directory containing video files from JSON.
    """

    def __init__(
        self,
        csv_path,
        json_path,
        num_frames=16,
        sample_fps=24,
        csv_root=None,
        json_root=None,
        sizes=[(512, 512), (384, 640), (640, 384)],
        ratios=[1/1, 3/5, 5/3],
    ):
        video_samples = []

        # Load from CSV
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            csv_list = list(reader)
        for vid in csv_list[1:]:  # no csv head
            vid_name = vid[0]
            vid_path = os.path.join(csv_root, vid_name)
            vid_caption = vid[1]
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])

        # Load from JSON
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for entry in json_data:
            vid_name = entry['vid']
            vid_path = os.path.join(json_root, f"{vid_name}.mp4")
            vid_caption = entry['caption']
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])

        self.samples = video_samples
        self.is_video = True
        self.sizes = sizes
        self.ratios = ratios
        self.num_frames = num_frames
        self.sample_fps = sample_fps
    
    def get_resize_size(self, orig_size, tgt_size):
        if (tgt_size[1]/tgt_size[0] - 1) * (orig_size[1]/orig_size[0] - 1) >= 0:
            alt_min = int(math.ceil(max(tgt_size)*min(orig_size)/max(orig_size)))
            resize_size = max(alt_min, min(tgt_size))
        else:
            alt_max = int(math.ceil(min(tgt_size)*max(orig_size)/min(orig_size)))
            resize_size = max(alt_max, max(tgt_size))
        return resize_size

    def get_closest_size(self, width, height):
        best_size_idx = np.argmin([abs(width/height-r) for r in self.ratios])
        return self.sizes[best_size_idx]
    
    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]
        
        video_capture = cv2.VideoCapture(path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            flag, frame = video_capture.read()
            if not flag:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)

        video_capture.release()
        sample_fps = self.sample_fps
        interval = max(int(fps / sample_fps), 1)
        frames = frames[::interval]

        if len(frames) < self.num_frames:
            num_frame_to_pack = self.num_frames - len(frames)
            recurrent_num = num_frame_to_pack // len(frames)
            frames = frames + recurrent_num * frames + frames[:(num_frame_to_pack % len(frames))]
            assert len(frames) >= self.num_frames, f'{len(frames)}'

        start_indexs = list(range(0, max(0, len(frames) - self.num_frames + 1)))            
        start_index = random.choice(start_indexs)

        filtered_frames = frames[start_index : start_index+self.num_frames]
        assert len(filtered_frames) == self.num_frames, f"The sampled frames should equals to {self.num_frames}"

        filtered_frames = torch.stack(filtered_frames).float() / 255
        height, width = filtered_frames.shape[2], filtered_frames.shape[3]
        
        size = self.get_closest_size(width, height)
        resize_size = self.get_resize_size((width, height), size)
        video_transform = get_transform(width, height, resize_size[0], resize_size[1], resize=True)

        filtered_frames = video_transform(filtered_frames)
        video = filtered_frames.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)

class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
        sizes=[(512, 512), (384, 640), (640, 384)],
        ratios=[1/1, 3/5, 5/3]
    ):
        video_samples = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            csv_list = list(reader)
        for vid in csv_list[1:]:  # no csv head
            vid_name = vid[0]
            vid_path = os.path.join(root, vid_name)
            vid_caption = vid[1]
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])
        self.samples = video_samples

        self.is_video = True
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.root = root
        self.sizes = sizes
        self.ratios = ratios

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]

        if self.is_video:
            is_exit = os.path.exists(path)
            if is_exit:
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                total_frames = len(vframes)
            else:
                total_frames = 0
            
            loop_index = index
            while(total_frames < self.num_frames or is_exit == False):
                loop_index += 1
                if loop_index >= len(self.samples):
                    loop_index = 0
                sample = self.samples[loop_index]
                path = sample[0]
                text = sample[1]

                is_exit = os.path.exists(path)
                if is_exit:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                    total_frames = len(vframes)
                else:
                    total_frames = 0
            #  video exits and total_frames >= self.num_frames
            
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
            
            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


class DatasetFromCSVAndJSON(torch.utils.data.Dataset):
    """Load video according to both csv and json files.

    Args:
        csv_path (str): Path to the CSV file.
        json_path (str): Path to the JSON file.
        num_frames (int): Number of video frames to load.
        frame_interval (int): Interval between frames.
        transform (callable): Transform to apply to video frames.
        csv_root (str): Root directory containing video files from CSV.
        json_root (str): Root directory containing video files from JSON.
    """

    def __init__(
        self,
        csv_path,
        json_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        csv_root=None,
        json_root=None
    ):
        video_samples = []

        # Load from CSV
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            csv_list = list(reader)
        for vid in csv_list[1:]:  # no csv head
            vid_name = vid[0]
            vid_path = os.path.join(csv_root, vid_name)
            vid_caption = vid[1]
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])

        # Load from JSON
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for entry in json_data:
            vid_name = entry['vid']
            vid_path = os.path.join(json_root, f"{vid_name}.mp4")
            vid_caption = entry['caption']
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])

        self.samples = video_samples
        self.is_video = True
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]

        if self.is_video:
            is_exit = os.path.exists(path)
            if is_exit:
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                total_frames = len(vframes)
            else:
                total_frames = 0
            
            loop_index = index
            while(total_frames < self.num_frames or is_exit == False):
                loop_index += 1
                if loop_index >= len(self.samples):
                    loop_index = 0
                sample = self.samples[loop_index]
                path = sample[0]
                text = sample[1]

                is_exit = os.path.exists(path)
                if is_exit:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                    total_frames = len(vframes)
                else:
                    total_frames = 0
            #  video exits and total_frames >= self.num_frames
            
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
            
            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)




def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_data_prompts(data_dir, video_size=(256,256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file)-1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2*idx]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
            image2 = Image.open(file_list[2*idx+1]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            _, filename = os.path.split(file_list[idx*2])
        else:
            image = Image.open(file_list[idx]).convert('RGB')
            image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)
        
    return filename_list, data_list, prompt_list