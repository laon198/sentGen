from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
model.load_state_dict(torch.load("/Users/laon/PycharmProjects/sentGen/model.pth"), map_location=torch.device('cpu'))
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

@api_view(["GET"])
def generate_sent(request):
    word = request.GET.get("word")
    char_list = word.split()
    res_sent = word

    # for char in char_list:
    #     res_sent += (" "+char)

    return Response(data=res_sent, status=status.HTTP_201_CREATED)