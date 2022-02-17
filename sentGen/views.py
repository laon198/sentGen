from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import random
import torch

model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
# model.load_state_dict(torch.load("/Users/laon/PycharmProjects/sentGen/model.pth", map_location=torch.device('cpu')))
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')


@api_view(["GET"])
def generate_sent(request):
    word = request.GET.get("word")
    char_list = list(word)
    res_sent = ""
    res_list = []

    # 생성
    for char in char_list:
        res_sent += char
        gen_ids = model.generate(torch.tensor([tokenizer.encode(res_sent)]),
                                 max_length=128,
                                 repetition_penalty=2.0,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 bos_token_id=tokenizer.bos_token_id,
                                 use_cache=True,
                                 do_sample=True,
                                 top_p=0.7)
        tmp_sent = tokenizer.decode(gen_ids[0, :].tolist())
        tmp_sent = " ".join(tmp_sent.split(" ")[0:len(res_sent.split(" ")) + random.randrange(1, 10)])
        hang = " ".join(tmp_sent.split(" ")[len(res_sent.split(" ")) - 1:])
        res_list.append(hang)
        res_sent = (tmp_sent + " ")

    print(res_list)

    return Response(data=res_list, status=status.HTTP_201_CREATED)
