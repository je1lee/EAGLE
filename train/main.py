import argparse
parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/ssd0/data/fast-llm/Llama-2-70B-Chat-fp16')
parser.add_argument('--configpath', type=str, default="/home/jewon/code/eagle_medusa/EAGLE/train/llama_2_chat_70B_config.json")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=3)
parser.add_argument('--gradient-accumulation-steps', type=int, default=11)
parser.add_argument('--tmpdir', type=str, default='/ssd0/data/fast-llm/eagle_train_data')
parser.add_argument('--outdir', type=str, default='/ssd0/checkpoints/fast-llm/with_val_epoch5')
parser.add_argument('--cpdir', type=str, default='/ssd0/checkpoints/fast-llm/with_val_epoch5')
args = parser.parse_args()


import sys
sys.path.append("../")

train_config={
    "lr":args.lr,
    "bs":args.bs,
    "gradient_accumulation_steps":args.gradient_accumulation_steps,
    "datapath":f"{args.tmpdir}",
    "is_warmup":True,
    "num_epochs":5,
    "num_warmup_steps":8000,
    "total_steps": 108641,
    "p_w":0.1,
    "v_w":1.0,
    "head_w":0.1,
    "num_workers":16,
    "embeding":True,
    "act":"No",
    "data_noise":True,
    "noise":"uniform",
    "mean":0.0,
    "std":0.2,
    "residual":"true,norm",
    "max_len":2048,
    "config_path":args.configpath,
    "b1":0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
}
import json
from safetensors import safe_open
#from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import peft

set_seed(0)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"], 
                          kwargs_handlers=[ddp_kwargs]
                          )
from model.cnets import Model
from model.configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase,get_linear_schedule_with_warmup,AutoConfig, get_cosine_schedule_with_warmup


if accelerator.is_main_process:
    # import wandb
    # wandb.init(project="ess", entity="yuhui-li",config=train_config)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"{args.outdir}/logs")



baseconfig=AutoConfig.from_pretrained(args.basepath)

head=torch.nn.Linear(baseconfig.hidden_size,baseconfig.vocab_size,bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False

from transformers.utils import PaddingStrategy

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor=data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"]=noisy_tensor
        return data

class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor=data["hidden_state_big"]
        noise = (torch.rand_like(tensor) -0.5)*self.std*512/tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"]=noisy_tensor
        return data

class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data=datapath
        self.transform = transform


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        # try:
        data=torch.load(self.data[index])
        new_data={}
        hidden_state=data['hidden_state'][:train_config["max_len"]][None,:]
        input_ids = data['input_ids'][:train_config["max_len"]][None,:]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None,:]

        # except:
        #     with open("error_path.txt", "w") as file:
        #         file.write(self.data[index])
        #     print('error path',self.data[index])


        length=hidden_state.shape[1]
        #length_q = data['query_ids'].shape[1]
        attention_mask=[1]*length
        loss_mask=loss_mask[0].tolist()
        loss_mask[-1]=0

        input_ids_target=input_ids[:,1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)


        target=hidden_state[:,1:,:]
        zeropadding=torch.zeros(1, 1, target.shape[2])
        target=torch.cat((target,zeropadding), dim=1)
        loss_mask[-1]=0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"]=target
        new_data["hidden_state_big"]=hidden_state
        new_data["input_ids"] = input_ids_target
        #sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        #label = data['y']

        if self.transform:
            new_data = self.transform(new_data)

        return new_data




class DataCollatorWithPadding:


    def paddingtensor(self,intensors,N):
        B,n,S=intensors.shape
        #padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self,intensors,N):
        B,n=intensors.shape
        padding_tensor = torch.zeros(B, N - n,dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states=torch.cat([self.paddingtensor(item['hidden_state_big'],max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor([item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids":batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target":batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

def top_accuracy(output, target, topk=(1, )):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

@torch.no_grad()
def getkacc(model,data,head,max_length=5):
    hidden_states=data["hidden_states"]
    input_ids=data["input_ids"]
    #attention_mask=data["attention_mask"]
    loss_mask=data["loss_mask"]
    #sample_mask=data["sample_mask"]
    target=data["target"]
    total=[0 for _ in range(max_length)]
    correct=[0 for _ in range(max_length)]
    bs,sl=hidden_states.shape[0],hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout=head(hidden_states)

    for i in range(bs):
        for j in range(sl):

            single_hidden_states=hidden_states[i,:j]
            single_input_ids=input_ids[i,:j]


            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1]-1] == 0:
                    break
                tmp_in_target_headout = hidden_states_headout[i,single_hidden_states.shape[1]-1]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1]-1]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token=input_ids[i,single_hidden_states.shape[1]-1]
                #tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                if not (target_in_token==tmp_token):
                    break
                if k == 0:
                    model.module.base_model.layers[0].self_attn.q_proj.active_switch = None
                    model.module.base_model.layers[0].self_attn.k_proj.active_switch = None
                else:
                    model.module.base_model.layers[0].self_attn.q_proj.active_switch = f"lora_{k}"
                    model.module.base_model.layers[0].self_attn.k_proj.active_switch = f"lora_{k}" 
                out_hidden = model(single_hidden_states, input_ids=single_input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                total[k] += 1
                if token==target_out_token:
                    correct[k]+=1
                else:
                    for kk in range(k+1,max_length):
                        total[kk]+=1
                    break

                single_hidden_states=torch.cat((single_hidden_states,out_hidden[:,-1:]),dim=1)
                single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)


    acc=[correct[i]/total[i] for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"]=="uniform":
        aug=AddUniformNoise(std=train_config["std"])
    else:
        aug=AddGaussianNoise(mean=train_config["mean"],std=train_config["std"])
else:
    aug=None

datapath=list_files(train_config["datapath"])

traindatapath=datapath[:int(len(datapath)*0.95)]
testdatapath=datapath[int(len(datapath)*0.95):]
# print('td',train_config["datapath"])
# print(datapath)
# exit()
traindataset=CustomDataset(traindatapath,transform=aug)
testdataset=CustomDataset(testdatapath)
train_loader=DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,collate_fn=DataCollatorWithPadding(),num_workers=train_config["num_workers"],pin_memory=True)
test_loader=DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,collate_fn=DataCollatorWithPadding(),num_workers=train_config["num_workers"],pin_memory=True)
# for batch_data in train_loader:
#     print(batch_data)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)


config=EConfig.from_pretrained(train_config["config_path"])
model=Model(config,load_emb=True,path=args.basepath)
weight_path = "/ssd0/data/fast-llm/eagle_train_reproduction/model.safetensors"
import safetensors
safetensors.torch.load_model(model, weight_path)

lora_config = peft.LoraConfig(
        lora_alpha = 16,
        r=8,
        lora_dropout=0.1,
        bias="none",
        target_modules= [
            "q_proj",
            "k_proj",
            ],
        task_type="CAUSAL_LM",
    )

# if hasattr(model, "enable_input_require_grads"):
#     model.enable_input_require_grads()
# else:
#     def make_inputs_require_grad(module, input, output):
#          output.requires_grad_(True)

#     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


model = peft.PeftModel(model, lora_config, "lora_1")
model.add_adapter("lora_2", lora_config)
model.add_adapter("lora_3", lora_config)
model.add_adapter("lora_4", lora_config)
# model_1 = peft.PeftModel(model, lora_config, "lora_2")
# model_2 = peft.PeftModel(model, lora_config, "lora_3")
# model_3 = peft.PeftModel(model, lora_config, "lora_4")


    # model.base_model.model.layers[0].self_attn.k_proj.lora_embedding_A.requires_grad_(True)
    # model.base_model.model.layers[0].self_attn.k_proj.lora_embedding_B.requires_grad_(True)

# model.base_model.model.layers[0].self_attn.q_proj.lora_embedding_A.requires_grad_(True)
# model.base_model.model.layers[0].self_attn.q_proj.lora_embedding_B.requires_grad_(True)
# model.base_model.model.layers[0].self_attn.k_proj.lora_embedding_A.requires_grad_(True)
# model.base_model.model.layers[0].self_attn.k_proj.lora_embedding_B.requires_grad_(True)

# if accelerator.is_main_process:
    # import wandb
    # wandb.init(project="ess", entity="yuhui-li",config=train_config)
    # model.save_pretrained("./temp")


# def switch_on(adapter_name):
#     model.base_model.model.layers[0].self_attn.q_proj.lora_dropout[adapter_name].requires_grad_(True)
#     model.base_model.model.layers[0].self_attn.q_proj.lora_A[adapter_name].requires_grad_(True)
#     model.base_model.model.layers[0].self_attn.q_proj.lora_B[adapter_name].requires_grad_(True)
#     # model.base_model.model.layers[0].self_attn.q_proj.lora_embedding_A.requires_grad_(True)
#     # model.base_model.model.layers[0].self_attn.q_proj.lora_embedding_B.requires_grad_(True)
    
#     model.base_model.model.layers[0].self_attn.k_proj.lora_dropout[adapter_name].requires_grad_(True)
#     model.base_model.model.layers[0].self_attn.k_proj.lora_A[adapter_name].requires_grad_(True)
#     model.base_model.model.layers[0].self_attn.k_proj.lora_B[adapter_name].requires_grad_(True)
#     # model.base_model.model.layers[0].self_attn.k_proj.lora_embedding_A.requires_grad_(True)
#     # model.base_model.model.layers[0].self_attn.k_proj.lora_embedding_B.requires_grad_(True)

criterion=nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"],train_config["b2"]))

num_epochs=train_config["num_epochs"]
num_warmup_steps=train_config["num_warmup_steps"]
total_steps=train_config["total_steps"]
is_warmup=train_config["is_warmup"]

if is_warmup:
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
    #                                             num_training_steps=total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, head, optimizer, train_loader,test_loader, scheduler = accelerator.prepare(
     model, head, optimizer, train_loader,test_loader, scheduler
    )
    for adapter_name in model.module.base_model.model.layers[0].self_attn.q_proj.lora_dropout.keys():
        if adapter_name == "lora_1":
            continue
        # model.base_model.model.layers[0].self_attn.q_proj.lora_dropout[adapter_name].requires_grad_(True)
        model.module.base_model.model.layers[0].self_attn.q_proj.lora_A[adapter_name].requires_grad_(True)
        model.module.base_model.model.layers[0].self_attn.q_proj.lora_B[adapter_name].requires_grad_(True)
        # model.base_model.model.layers[0].self_attn.q_proj.lora_embedding_A.requires_grad_(True)
        # model.base_model.model.layers[0].self_attn.q_proj.lora_embedding_B.requires_grad_(True)
        
        # model.base_model.model.layers[0].self_attn.k_proj.lora_dropout[adapter_name].requires_grad_(True)
        model.module.base_model.model.layers[0].self_attn.k_proj.lora_A[adapter_name].requires_grad_(True)
        model.module.base_model.model.layers[0].self_attn.k_proj.lora_B[adapter_name].requires_grad_(True)
else:
    model,head, optimizer, train_loader,test_loader = accelerator.prepare(
     model,head, optimizer, train_loader,test_loader
    )

# accelerator.load_state("/home/jewon/code/FASTLLM/EAGLE/train/test/state_1")

test_flag = True

global_step=0
for epoch in range(num_epochs):
    top_3acc=[0 for _ in range(3)]
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    
    total = 0
    epoch_loss=0
    num_batches=0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if batch_idx==2425 or batch_idx==2426 or batch_idx==2424:
            continue
        optimizer.zero_grad()
        with torch.no_grad():
            model.module.base_model.layers[0].self_attn.q_proj.active_switch = None
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = None
            predict_0 = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            if test_flag:
                _predict_1 = model(predict_0[:,1:,:], input_ids=data["input_ids"][..., 1:], attention_mask=data["attention_mask"][..., 1:])
                _predict_2 = model(_predict_1[:,1:,:], input_ids=data["input_ids"][..., 2:], attention_mask=data["attention_mask"][..., 2:])
                _predict_3 = model(_predict_2[:,1:,:], input_ids=data["input_ids"][..., 3:], attention_mask=data["attention_mask"][..., 3:])
        if test_flag:
            # model.module.set_adapter("lora_1")
            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_1"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_1"
            predict_1 = model(predict_0[:,1:,:], input_ids=data["input_ids"][..., 1:], attention_mask=data["attention_mask"][..., 1:])
            # model.module.set_adapter("lora_2")
            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_2"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_2"
            predict_2 = model(_predict_1[:,1:,:], input_ids=data["input_ids"][..., 2:], attention_mask=data["attention_mask"][..., 2:])
            # model.module.set_adapter("lora_3")
            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_3"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_3"
            predict_3 = model(_predict_2[:,1:,:], input_ids=data["input_ids"][..., 3:], attention_mask=data["attention_mask"][..., 3:])
            # model.module.set_adapter("lora_4")
            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_4"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_4"
            predict_4 = model(_predict_3[:,1:,:], input_ids=data["input_ids"][..., 4:], attention_mask=data["attention_mask"][..., 4:])
        else:    
            model.module.set_adapter("lora_1")
            predict_1 = model(predict_0[:,1:,:], input_ids=data["input_ids"][..., 1:], attention_mask=data["attention_mask"][..., 1:])
            model.module.set_adapter("lora_2")
            predict_2 = model(predict_1[:,1:,:], input_ids=data["input_ids"][..., 2:], attention_mask=data["attention_mask"][..., 2:])
            model.module.set_adapter("lora_3")
            predict_3 = model(predict_2[:,1:,:], input_ids=data["input_ids"][..., 3:], attention_mask=data["attention_mask"][..., 3:])
            # model.module.set_adapter("lora_4")
            # predict_4 = model(predict_3[:,1:,:], input_ids=data["input_ids"][..., 4:], attention_mask=data["attention_mask"][..., 4:])
    
        # print("hello")
        with torch.no_grad():
            target_head = head(data["target"])
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()
        # out_head_0 = head(predict_0)
        # out_logp_0 = nn.LogSoftmax(dim=2)(out_head_0)
        out_head_1 = head(predict_1)
        out_logp_1 = nn.LogSoftmax(dim=2)(out_head_1)
        out_head_2 = head(predict_2)
        out_logp_2 = nn.LogSoftmax(dim=2)(out_head_2)
        out_head_3 = head(predict_3)
        out_logp_3 = nn.LogSoftmax(dim=2)(out_head_3)
        out_head_4 = head(predict_4)
        out_logp_4 = nn.LogSoftmax(dim=2)(out_head_4)
        

        loss_mask = data["loss_mask"][:, :, None]
        # plogp = target_p * out_logp_0
        plogp_1 = target_p[:,1:,:] * out_logp_1
        plogp_2 = target_p[:,2:,:] * out_logp_2
        plogp_3 = target_p[:,3:,:] * out_logp_3
        plogp_4 = target_p[:,4:,:] * out_logp_4
        
        ploss_1 = -torch.sum(torch.sum(loss_mask[:,1:,:] * plogp_1, 2)) / loss_mask[:,1:,:].sum()
        ploss_2 = -torch.sum(torch.sum(loss_mask[:,2:,:] * plogp_2, 2)) / loss_mask[:,2:,:].sum()
        ploss_3 = -torch.sum(torch.sum(loss_mask[:,3:,:] * plogp_3, 2)) / loss_mask[:,3:,:].sum()
        ploss_4 = -torch.sum(torch.sum(loss_mask[:,4:,:] * plogp_4, 2)) / loss_mask[:,4:,:].sum()
        
        if not test_flag:
            ploss = ploss_1 + 0.5*ploss_2 + 0.25*ploss_3 + 0.125*ploss_4
            # ploss = ploss_1 + 0.5*ploss_2 + 0.25*ploss_3
        else:
            ploss = ploss_1 + ploss_2 + ploss_3 + ploss_4
            # ploss = ploss_1 + ploss_2 + ploss_3
            
        # ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / loss_mask.sum()
        vloss_1 = criterion(predict_1, data["target"][:,1:,:])
        vloss_1 = torch.sum(torch.mean(loss_mask[:,1:,:] * vloss_1, 2)) / loss_mask[:,1:,:].sum()
        vloss_2 = criterion(predict_2, data["target"][:,2:,:])
        vloss_2 = torch.sum(torch.mean(loss_mask[:,2:,:] * vloss_2, 2)) / loss_mask[:,2:,:].sum()
        vloss_3 = criterion(predict_3, data["target"][:,3:,:])
        vloss_3 = torch.sum(torch.mean(loss_mask[:,3:,:] * vloss_3, 2)) / loss_mask[:,3:,:].sum()
        vloss_4 = criterion(predict_4, data["target"][:,4:,:])
        vloss_4 = torch.sum(torch.mean(loss_mask[:,4:,:] * vloss_4, 2)) / loss_mask[:,4:,:].sum()
        
        if not test_flag:
            vloss = vloss_1 + 0.5*vloss_2 + 0.25*vloss_3 + 0.125*vloss_4
            # vloss = vloss_1 + 0.5*vloss_2 + 0.25*vloss_3
        else:
            vloss = vloss_1 + vloss_2 + vloss_3 + vloss_4
            # vloss = vloss_1 + vloss_2 + vloss_3
        
        # vloss = criterion(predict_0, data["target"])
        # vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / loss_mask.sum()
        loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
        # print(loss)
        accelerator.backward(loss)
        accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
        optimizer.step()
        global_step+=1

        if loss!=loss and accelerator.is_main_process:
            print(f"nan, Epoch {epoch}, batch id {batch_idx}")
            with open('nan.txt','w') as f:
                f.write(f"nan, Epoch {epoch}, batch id {batch_idx}")
                torch.save(data,'nandata.ckpt')
            exit()

        if is_warmup:
            scheduler.step()

        with torch.no_grad():
            # _, predicted = torch.max(out_head, 2)
            _, predicted_1 = torch.max(out_head_1, 2)
            _, predicted_2 = torch.max(out_head_2, 2)
            _, predicted_3 = torch.max(out_head_3, 2)
            _, predicted_4 = torch.max(out_head_4, 2)
            
            _, target = torch.max(target_head, 2)
            ct=loss_mask.sum().item()
            cc_1=((predicted_1 == target[:,1:])*loss_mask[:,1:,:].squeeze()).sum().item()
            cc_2=((predicted_2 == target[:,2:])*loss_mask[:,2:,:].squeeze()).sum().item()
            cc_3=((predicted_3 == target[:,3:])*loss_mask[:,3:,:].squeeze()).sum().item()
            cc_4=((predicted_4 == target[:,4:])*loss_mask[:,4:,:].squeeze()).sum().item()
            
            # out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            # target = target.view(-1)[loss_mask.view(-1) == 1]
            # topkacc = top_accuracy(out_head, target,(1,2,3))
            # for top_i in range(len(topkacc)):
            #     top_3acc[top_i]+=topkacc[top_i]
            total += ct
            correct_1 += cc_1
            correct_2 += cc_2
            correct_3 += cc_3
            correct_4 += cc_4
            
        if accelerator.is_main_process and ct!=0:
            logdict={"train/lr":optimizer.optimizer.param_groups[0]["lr"],
                     "train/vloss_1":vloss_1.item(), "train/vloss_2":vloss_2.item(), "train/vloss_3":vloss_3.item(),
                     "train/vloss_4":vloss_4.item(),
                     "train/vloss":vloss.item(),
                     "train/ploss_1":ploss_1.item(), "train/ploss_2":ploss_2.item(), "train/ploss_3":ploss_3.item(), 
                     "train/ploss_4":ploss_4.item(),
                     "train/ploss":ploss.item(),
                     "train/loss":loss.item(),"train/acc_1":cc_1/ct, "train/acc_2":cc_2/(ct-1), "train/acc_3":cc_3/(ct-2), 
                     "train/acc_4":cc_4/(ct-3)
                     }
            # for id,i in enumerate(top_3acc):
            #     logdict[f'train/top_{id+1}_acc']=topkacc[id].item()/ct
            writer.add_scalars("train_metrics",logdict,global_step=global_step)
            # for id,i in enumerate(top_3acc):
            #     writer.add_scalars(f"train/top_{id+1}_acc", {f"train/top_{id+1}_acc":topkacc[id].item()/ct},global_step=global_step)
        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    # correct,total=torch.tensor(correct).cuda(),torch.tensor(total).cuda()
    # correct, total = accelerator.gather_for_metrics((correct, total))
    # correct, total=correct.sum().item(), total.sum().item()
    # epoch_loss /= num_batches
    # top_3acc = accelerator.gather_for_metrics(top_3acc)
    # if accelerator.is_local_main_process:
    #     for id, i in enumerate(top_3acc):
    #         pass
            # wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    # if accelerator.is_local_main_process:
    #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
    #     # print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
    #     # wandb.log({"train/epochacc":correct / total,"train/epochloss":epoch_loss})
    #     writer.add_scalars("train_metrics", {"train/epochacc":correct / total,"train/epochloss":epoch_loss},global_step=global_step)
    
    
    if accelerator.is_local_main_process:
        accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
        model.save_pretrained(f"{args.cpdir}/lora_adapters_{epoch}")

    top_3acc = [0 for _ in range(3)]
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.eval()

    k_acc=[[]for i in range(5)]
    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            if batch_idx<10:
                acces=getkacc(model,data,head,max_length=5)
                for i in range(len(acces)):
                    k_acc[i].append(acces[i])
            ###        

            model.module.base_model.layers[0].self_attn.q_proj.active_switch = None
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = None
            predict_0 = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            _predict_1 = model(predict_0[:,1:,:], input_ids=data["input_ids"][..., 1:], attention_mask=data["attention_mask"][..., 1:])
            _predict_2 = model(_predict_1[:,1:,:], input_ids=data["input_ids"][..., 2:], attention_mask=data["attention_mask"][..., 2:])
            _predict_3 = model(_predict_2[:,1:,:], input_ids=data["input_ids"][..., 3:], attention_mask=data["attention_mask"][..., 3:])


            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_1"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_1"
            predict_1 = model(predict_0[:,1:,:], input_ids=data["input_ids"][..., 1:], attention_mask=data["attention_mask"][..., 1:])

            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_2"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_2"
            predict_2 = model(_predict_1[:,1:,:], input_ids=data["input_ids"][..., 2:], attention_mask=data["attention_mask"][..., 2:])

            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_3"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_3"
            predict_3 = model(_predict_2[:,1:,:], input_ids=data["input_ids"][..., 3:], attention_mask=data["attention_mask"][..., 3:])

            model.module.base_model.layers[0].self_attn.q_proj.active_switch = "lora_4"
            model.module.base_model.layers[0].self_attn.k_proj.active_switch = "lora_4"
            predict_4 = model(_predict_3[:,1:,:], input_ids=data["input_ids"][..., 4:], attention_mask=data["attention_mask"][..., 4:])        
            
            target_head = head(data["target"])
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()
            # out_head_0 = head(predict_0)
            # out_logp_0 = nn.LogSoftmax(dim=2)(out_head_0)
            out_head_1 = head(predict_1)
            out_logp_1 = nn.LogSoftmax(dim=2)(out_head_1)
            out_head_2 = head(predict_2)
            out_logp_2 = nn.LogSoftmax(dim=2)(out_head_2)
            out_head_3 = head(predict_3)
            out_logp_3 = nn.LogSoftmax(dim=2)(out_head_3)
            out_head_4 = head(predict_4)
            out_logp_4 = nn.LogSoftmax(dim=2)(out_head_4)
            

            loss_mask = data["loss_mask"][:, :, None]
            # plogp = target_p * out_logp_0
            plogp_1 = target_p[:,1:,:] * out_logp_1
            plogp_2 = target_p[:,2:,:] * out_logp_2
            plogp_3 = target_p[:,3:,:] * out_logp_3
            plogp_4 = target_p[:,4:,:] * out_logp_4
            
            ploss_1 = -torch.sum(torch.sum(loss_mask[:,1:,:] * plogp_1, 2)) / loss_mask[:,1:,:].sum()
            ploss_2 = -torch.sum(torch.sum(loss_mask[:,2:,:] * plogp_2, 2)) / loss_mask[:,2:,:].sum()
            ploss_3 = -torch.sum(torch.sum(loss_mask[:,3:,:] * plogp_3, 2)) / loss_mask[:,3:,:].sum()
            ploss_4 = -torch.sum(torch.sum(loss_mask[:,4:,:] * plogp_4, 2)) / loss_mask[:,4:,:].sum()
            
            if not test_flag:
                ploss = ploss_1 + 0.5*ploss_2 + 0.25*ploss_3 + 0.125*ploss_4
                # ploss = ploss_1 + 0.5*ploss_2 + 0.25*ploss_3
            else:
                ploss = ploss_1 + ploss_2 + ploss_3 + ploss_4
                # ploss = ploss_1 + ploss_2 + ploss_3
                
            # ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / loss_mask.sum()
            vloss_1 = criterion(predict_1, data["target"][:,1:,:])
            vloss_1 = torch.sum(torch.mean(loss_mask[:,1:,:] * vloss_1, 2)) / loss_mask[:,1:,:].sum()
            vloss_2 = criterion(predict_2, data["target"][:,2:,:])
            vloss_2 = torch.sum(torch.mean(loss_mask[:,2:,:] * vloss_2, 2)) / loss_mask[:,2:,:].sum()
            vloss_3 = criterion(predict_3, data["target"][:,3:,:])
            vloss_3 = torch.sum(torch.mean(loss_mask[:,3:,:] * vloss_3, 2)) / loss_mask[:,3:,:].sum()
            vloss_4 = criterion(predict_4, data["target"][:,4:,:])
            vloss_4 = torch.sum(torch.mean(loss_mask[:,4:,:] * vloss_4, 2)) / loss_mask[:,4:,:].sum()
            


            vloss = vloss_1 + vloss_2 + vloss_3 + vloss_4
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss

            if loss!=loss and accelerator.is_main_process:
                print(f"nan, Epoch {epoch}, batch id {batch_idx}")
                with open('nan.txt','w') as f:
                    f.write(f"nan, Epoch {epoch}, batch id {batch_idx}")
                    torch.save(data,'nandata.ckpt')
                exit()


            # _, predicted = torch.max(out_head, 2)
            _, predicted_1 = torch.max(out_head_1, 2)
            _, predicted_2 = torch.max(out_head_2, 2)
            _, predicted_3 = torch.max(out_head_3, 2)
            _, predicted_4 = torch.max(out_head_4, 2)
            
            _, target = torch.max(target_head, 2)
            ct=loss_mask.sum().item()
            cc_1=((predicted_1 == target[:,1:])*loss_mask[:,1:,:].squeeze()).sum().item()
            cc_2=((predicted_2 == target[:,2:])*loss_mask[:,2:,:].squeeze()).sum().item()
            cc_3=((predicted_3 == target[:,3:])*loss_mask[:,3:,:].squeeze()).sum().item()
            cc_4=((predicted_4 == target[:,4:])*loss_mask[:,4:,:].squeeze()).sum().item()
            
            # out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            # target = target.view(-1)[loss_mask.view(-1) == 1]
            # topkacc = top_accuracy(out_head, target,(1,2,3))
            # for top_i in range(len(topkacc)):
            #     top_3acc[top_i]+=topkacc[top_i]
            total += ct
            correct_1 += cc_1
            correct_2 += cc_2
            correct_3 += cc_3
            correct_4 += cc_4
                
            if accelerator.is_main_process and ct!=0:
                logdict={"test/vloss_1":vloss_1.item(), "test/vloss_2":vloss_2.item(), "test/vloss_3":vloss_3.item(),
                        "test/vloss_4":vloss_4.item(),
                        "test/vloss":vloss.item(),
                        "test/ploss_1":ploss_1.item(), "test/ploss_2":ploss_2.item(), "test/ploss_3":ploss_3.item(), 
                        "test/ploss_4":ploss_4.item(),
                        "test/ploss":ploss.item(),
                        "test/loss":loss.item(),"test/acc_1":cc_1/ct, "test/acc_2":cc_2/(ct-1), "test/acc_3":cc_3/(ct-2), 
                        "test/acc_4":cc_4/(ct-3)
                        }
                # for id,i in enumerate(top_3acc):
                #     logdict[f'train/top_{id+1}_acc']=topkacc[id].item()/ct
                writer.add_scalars("train_metrics",logdict,global_step=global_step)
                # for id,i in enumerate(top_3acc):
                #     writer.add_scalars(f"train/top_{id+1}_acc", {f"train/top_{id+1}_acc":topkacc[id].item()/ct},global_step=global_step)

                
                
                    
            ###
            # predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            # target_head = head(data["target"])
            # target_p = nn.Softmax(dim=2)(target_head)
            # target_p = target_p.detach()
            # out_head = head(predict)
            # out_logp = nn.LogSoftmax(dim=2)(out_head)
            # loss_mask = data["loss_mask"][:, :, None]
            # plogp = target_p * out_logp
            # ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / loss_mask.sum()
            # vloss = criterion(predict, data["target"])
            # vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / loss_mask.sum()
            # loss=train_config["v_w"]*vloss+train_config["p_w"]*ploss
            # _, predicted = torch.max(out_head, 2)
            # _, target = torch.max(target_head, 2)
            # ct = loss_mask.sum().item()
            # cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            # out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            # target = target.view(-1)[loss_mask.view(-1) == 1]
            # topkacc = top_accuracy(out_head, target, (1, 2, 3))
            # for top_i in range(len(topkacc)):
            #     top_3acc[top_i] += topkacc[top_i]
            # total += ct
            # correct += cc
        epoch_loss += loss.item()
        num_batches += 1

    mean_acces=[]
    for id,i in enumerate(k_acc):
        mean_acc=np.array(i).mean()
        mean_acc=torch.tensor(mean_acc).cuda()
        mean_acces.append(mean_acc)

    mean_acces=accelerator.gather_for_metrics(mean_acces)
    if accelerator.is_local_main_process:
        for id,i in enumerate(mean_acces):
            mean_acc=i.mean().item()
            # wandb.log({f"test/{id}_acc":mean_acc})
            writer.add_scalars(f"test/{id}_acc", {f"test/{id}_acc":mean_acc},global_step=epoch)


    # correct,total=torch.tensor(correct).cuda(),torch.tensor(total).cuda()
    # correct, total = accelerator.gather_for_metrics((correct, total))
    # correct, total=correct.sum().item(), total.sum().item()
    # top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            # wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
            writer.add_scalars(f'test/top_{id + 1}_acc', {f'test/top_{id + 1}_acc': i.sum().item() / total},global_step=epoch)

    # epoch_loss /= num_batches
    if accelerator.is_local_main_process:
        # print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        # print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        # wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
        writer.add_scalars("test_metrics", {"test/epochacc": correct / total, "test/epochloss": epoch_loss},global_step=epoch)
        #accelerator.save_model(model, f"checkpoints/model_{epoch}")
        # accelerator.save_state(output_dir=f"{args.outdir}/state_{epoch}")
        # os.system(f"cp -r {args.outdir} {args.cpdir}")
        accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
        model.save_pretrained(f"{args.cpdir}/lora_adapters_{epoch}")
        

if accelerator.is_main_process:
    # import wandb
    # wandb.init(project="ess", entity="yuhui-li",config=train_config)
    writer.close()
