# -*- coding:utf-8 -*-
import tornado.ioloop
import tornado.web
import argparse
import traceback
import logging
import json
import torch
from transformers.generation.utils import GenerationConfig
from models.modeling_baichuan2 import BaichuanForCausalLM
from transformers import AutoTokenizer, AutoModel

class BatchChat(tornado.web.RequestHandler):
    def post(self):
        try:
            param_json = json.loads(self.request.body.decode('utf-8'))
            requestId = param_json.get("requestId")#batchInput["",""]
            token = param_json.get("token")
            biz = param_json.get("biz")
            batchInput = biz.get("extra").get("batchInput")#batchInput["",""]
            prompt1=batchInput[0]
            prompt2=batchInput[1]
            curAnswer = baichuan_service.generate_batch(prompt1,prompt2)

            result = {
                "status": 0,
                "msg": "success",
                "data": curAnswer
            }
            self.finish(result)
        except Exception as e:
            # logging.error("GetComment {}".format(e))
            traceback.print_exc()
            # torch.cuda.empty_cache()
            self.finish({
                "status": 1,
                "msg": "error: {}".format(e),
                "data":""
            })
class SingleChat(tornado.web.RequestHandler):
    def post(self):
        try:
            param_json = json.loads(self.request.body.decode('utf-8'))
            prompt = param_json.get("prompt")#batchInput["",""]
            curAnswer = baichuan_service.generate_one(prompt)

            result = {
                "status": 0,
                "message": "success",
                "newAnswer": curAnswer
            }
            self.finish(result)
        except Exception as e:
            logging.error("GetComment {}".format(e))
            traceback.print_exc()
            self.finish({
                "status": 1,
                "message": "error: oom"
            })

def make_app():
    return tornado.web.Application([
        (r"/pulitzer/mcu/generate/baichuan", BatchChat),
        (r"/mcu/api/generate/baichuan/singleChat", SingleChat),
    ])

class BaichuanService(object):
    def __init__(self,):
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/mcu/public/plms/baichuan2-7b/", use_fast=False,
                                                  trust_remote_code=True)
        self.model = BaichuanForCausalLM.from_pretrained("/mnt/mcu/public/plms/baichuan2-7b/", device_map={"": "cpu"},
                                                    torch_dtype=torch.float16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained("/mnt/mcu/public/plms/baichuan2-7b/")
        # self.model.half()
        self.model.to("cuda:{}".format(args.gpu))
        self.model.eval()
        # self.model.half().to(self.device)

    def generate_batch(self, prompt1,prompt2):
        messages_p1=[{"role": "user", "content": prompt1}]
        messages_p2=[{"role": "user", "content": prompt2}]
        response = self.model.chat(self.tokenizer, messages_p1, messages_p2)
        return response
    def generate_one(self, prompt1):
        messages_p1=[{"role": "user", "content": prompt1}]
        response = self.model.single_chat(self.tokenizer, messages_p1)
        return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configPath", type=str, default="./config/config.ini")
    parser.add_argument('--maxlen', type=str, default=1500)
    parser.add_argument('--port', type=str, default="11678")
    parser.add_argument("--gpu", type=str, default="7")

    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()
    port = args.port
    # logging.info("mcu-comment API start in port {}".format(port))

    baichuan_service = BaichuanService()
    app = make_app()
    app.listen(port)
    print("start")
    tornado.ioloop.IOLoop.current().start()
