import math
import os
from timechat.datasets.datasets.base_dataset import BaseDataset
from timechat.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
from timechat.processors import transforms_video, AlproVideoTrainProcessor
from torchvision import transforms
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
from timechat.conversation.conversation_video import Conversation, SeparatorStyle

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
video_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
llama_v2_video_conversation = Conversation(
    system=" ",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
IGNORE_INDEX = -100


class Video_Instruct_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root, num_video_query_token=32,
                 tokenizer_name='/mnt/workspace/ckpt/vicuna-13b/', data_type='video', model_type='vicuna', num_frm=8,
                 sample_type='rand', max_txt_len=512, stride=32):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = num_frm
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms=self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type
        self.sample_type = sample_type
        self.max_txt_len = max_txt_len
        self.stride = stride

    def _get_video_path(self, sample):
        rel_video_fp = sample['video']
        full_video_fp = os.path.join(self.vis_root, rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]

                video_path = self._get_video_path(sample)
                conversation_list = sample['QA']

                video, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling=self.sample_type, return_msg=True
                )
                video = self.transform(video)
                if 'cn' in self.data_type:
                    msg = ""
                # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0
                cur_n_frm = video.shape[1]
                cur_token_len = self.num_video_query_token * math.ceil(
                    cur_n_frm / self.stride) if self.stride > 0 else self.num_video_query_token
                sources = preprocess_multimodal(copy.deepcopy(conversation_list), None, cur_token_len=cur_token_len,
                                                msg=msg)
                new_sources = convert_source_vicuna_format(sources)
                if index == 0:
                    print(new_sources)
                if self.model_type == 'vicuna':
                    data_dict = preprocess(
                        new_sources,
                        self.tokenizer,
                        self.max_txt_len
                    )
                elif self.model_type == 'llama_v2':
                    data_dict = preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer,
                        self.max_txt_len
                    )
                else:
                    print('not support')
                    raise ('not support')
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = video
                # timestamp
                all_timestamps = msg.split('at')[1].replace('seconds.', '').strip().split(
                    ',')  # extract timestamps from msg
                all_timestamps = [f'This frame is sampled at {t.strip()} second.' for t in all_timestamps]
                all_timestamps = self.tokenizer(
                    all_timestamps,
                    return_tensors="pt",
                    padding="longest",
                    max_length=32,
                    truncation=True,
                )
                data_dict['timestamps'] = all_timestamps
            except:
                print(f"Failed to load examples with video: {video_path}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type": 'video',
            "timestamps": data_dict['timestamps'],
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, instances):
        input_ids, labels, timestamps = tuple([instance[key] for instance in instances]
                                              for key in ("text_input", "labels", "timestamps"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in
                   images):  # nb of frames of all videos is ${num_frm}
                batch['images'] = torch.stack(images)
                timestamps_input_ids, timestamps_attention_mask = [], []
                for timestamp in timestamps:
                    n_frm = timestamp['input_ids'].shape[0]
                    for i in range(n_frm):
                        timestamps_input_ids.append(timestamp['input_ids'][i])
                        timestamps_attention_mask.append(timestamp['attention_mask'][i])
                timestamps_input_ids = torch.nn.utils.rnn.pad_sequence(
                    timestamps_input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id)
                timestamps_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    timestamps_attention_mask,
                    batch_first=True,
                    padding_value=0)
                batch['timestamps'] = {'input_ids': timestamps_input_ids, 'attention_mask': timestamps_attention_mask}
            else:  # nb of frames of some videos is less than ${num_frm}
                print(f'image shape not match: {[x.shape for x in images]}')
                batch['images'] = images
                batch_timestamps = []
                for timestamp in timestamps:
                    batch_timestamps.append(
                        {'input_ids': timestamp['input_ids'], 'attention_mask': timestamp['attention_mask']})
                batch['timestamps'] = batch_timestamps
        batch['conv_type'] = 'multi'
        return batch


def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence['q']
            role_1_msg = sentence['a']
            new_source.append({
                'from': 'human',
                'value': role_0_msg,
            })
            new_source.append({
                'from': 'gpt',
                'value': role_1_msg,
            })
        new_sources.append(new_source)
    return new_sources


def preprocess_multimodal(
        conversation_list: Sequence[str],
        multimodal_cfg: dict,
        cur_token_len: int,
        msg=''
) -> Dict:
    # 将conversational list中
    is_multimodal = True
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    conversation_list[0]["q"] = "<Video>" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + "</Video> " + msg + \
                                conversation_list[0]["q"]
    return [conversation_list]


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_txt_len: int = 512, ) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_txt_len,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        max_txt_len: int,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer, max_txt_len)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer, max_txt_len)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_for_llama_v2(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        max_txt_len: int = 512,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    conv = copy.deepcopy(llama_v2_video_conversation.copy())
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for source in sources:
        # <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n
        header = f"<s>[INST] <<SYS>>\n{conv.system}\n</SYS>>\n\n"

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=max_txt_len,
        truncation=True,
    ).input_ids
    targets = copy.deepcopy(input_ids)

    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # 为什么减去2,speical token 的数目

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len
