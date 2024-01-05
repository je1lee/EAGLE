from model.ea_model import EaModel
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template

base_model_path = "/ssd0/data/fast-llm/Llama-2-13b-chat-fp16/"
EAGLE_model_path = "yuhuili/EAGLE-llama2-chat-13B"


# model = EaModel.from_pretrained(  
#     base_model_path=base_model_path,  
#     ea_model_path=EAGLE_model_path,  
#     torch_dtype=torch.float16,  
#     low_cpu_mem_usage=True,  
#     device_map="auto"  
# )

model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

model.eval()
conv = get_conversation_template("llama-2-chat")
sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
conv.system_message = sys_p
prompt="Hello Nice to meet you! give me a five story about human and AI"
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# input_ids=model.tokenizer([prompt]).input_ids
# input_ids = torch.as_tensor(input_ids).cuda()

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

start = time.time()
# output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output_ids = model.generate(input_ids, temperature=0.5, max_new_tokens=512)
end = time.time()
print("time cost: ", end - start)
print("time per token: ", (end - start) / (len(output_ids[0]) - len(input_ids[0])))
output=model.tokenizer.decode(output_ids[0])
print(output)