# cortex version 0.19 and may not deploy correctly on other releases of cortex
import torch
import boto3
import nltk
import os
import json
import onnxruntime as rt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import cstm_generate

def s3_download(cortex_base, modelpath, s3bucket, s3):
    ''' downloads folder and all subfiles from s3 folder'''
    #making path in cortex directory
    os.makedirs(os.path.join(cortex_base, modelpath), exist_ok=True)
    #getting response object for s3 folder
    response = s3.list_objects_v2(Bucket=s3bucket, Prefix=modelpath)

    #downloading objects from s3 folder using response object as guide
    for s3_obj in response["Contents"]:
        s3_obj_key = s3_obj["Key"]
        s3.download_file(s3bucket, s3_obj_key, os.path.join(cortex_base, s3_obj_key))
        #print so can see what is downloaded in the logs
        print("downloaded s3://{}/{} to {}/{}. \n".format(s3bucket, s3_obj_key, cortex_base, s3_obj_key), end = '')

def text_cleanup(text_gen, list_gen, mode):
    """Take text generated, Text generated as list of sentences and mode, returns string of finalized sentence. Sorts generated text in different
    cases"""
    #generated only 1 sentence
    if len(list_gen)<=1:
        clean_text_gen=text_gen
    else:
        #where completion[0] == punctuation
        if text_gen[0] not in [".", "?", "!"]:
            #getting final text for each mode
            if mode=='s-completion':
                clean_text_gen=list_gen[0]
            #if give complete sentence, then gives them two sentences
            elif mode=="s-completion+":
                clean_text_gen=list_gen[0]+list_gen[1]
            #paragraph completion
            else:
                line_break=text_gen.find("\n")
                if line_break !=-1:
                    clean_text_gen=text_gen[0:line_break-1]
                else:
                    clean_text_gen=text_gen
        else:
            if mode=='s-completion':
                clean_text_gen=list_gen[0]+list_gen[1]
            #if give complete sentence, then gives them two sentences
            elif mode=="s-completion+":
                #if not enough sentences
                if len(list_gen)<=2:
                    clean_text_gen=list_gen[0]+list_gen[1]
                else:
                    clean_text_gen=list_gen[0]+list_gen[1]+list_gen[2]
            else:
                line_break=text_gen.find("\n")
                if line_break !=-1:
                    clean_text_gen=text_gen[0:line_break-1]
                else:
                    clean_text_gen=text_gen

    return clean_text_gen

class PythonPredictor:
    def __init__(self, config):
        '''Runs once when deploying'''

        #sent_tokenize not included so have to download
        nltk.download("punkt")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")
        #credentials
        aws_access_key_id="AKIAWU7ZSW5FTHNASMNN"
        aws_secret_access_key="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"
        cortex_base1="tokenizer"
        modelpath1='transformers-tokenizer-gpt2'
        cortex_base2="models"
        modelpath2='onnx-124M-ArtOfTimeMachine-300steps'
        s3bucket='tunguska-generate-1'
        api_key_server='A34pDZUY7gHZORPnfDh7bgoO8Rt7pUS'

        #creating client and getting list of objects
        s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, \
                  aws_secret_access_key=aws_secret_access_key)
        #downloading motokenizer and model
        s3_download(cortex_base1, modelpath1, s3bucket, s3)
        s3_download(cortex_base2, modelpath2, s3bucket, s3)

        #for transformers select folder located in
        self.tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(cortex_base1, modelpath1))

        #for inference session have to select the model file
        self.model1 = rt.InferenceSession(os.path.join(cortex_base2, modelpath2, modelpath2+".onnx"))

        # if using torch model then uncomment this and 1 other spot
        #self.model1 = GPT2LMHeadModel.from_pretrained(\
        #os.path.join(cortex_base1, modelpath1), use_cache=True).to(self.device)


    def predict(self, payload):
        '''Is called each time you make a call to the predictor '''
        #for data logging
        print(payload)

        #prevent accidently someone trying to overload our parsing by passing logical statements
        api_key=str(payload["api_key"])

        #verifying key
        if api_key != api_key_server:
            final_answer={"text": "Invalid api_key", "truncated": "Invalid api_key"}

        else :
            #setting inputs for length
            if payload['mode']=='s-completion':
                gen_len=45
            elif payload['mode']=="s-completion+":
                gen_len=90
            else:
                gen_len=450

            #storing text for readibility and efficiency
            truncated=False
            raw_text=payload["text"]

            #tokenizing
            tokens = self.tokenizer.encode(raw_text, return_tensors="pt").to(self.device)
            #generating input length
            input_length = len(tokens[0])

            #if input is too long, then shortening so program will still run
            if input_length+gen_len > 1023:
                text = self.tokeizer.decode(tokens[-573:])
                tokens = tokens[-573:]
                input_length = 573
                truncated=True
            #do nothing to text, tokens
            else :
                text = raw_text

            if payload["pred_name"]=="model1":
                prediction = cstm_generate.generate_no_beam_search(self.model1, tokens,
                input_length,
                input_length+gen_len,
                input_length+gen_len-5,
                payload["temperature"],
                payload["top_k"],
                payload["top_p"],
                payload["repetition_penalty"],
                payload["batch_size"],
                sess_type="onnx"
                )


            #uncomment if using torch model
            #if payload["pred_name"]=="model1":
            #    prediction = self.model1.generate(tokens, min_length=input_length+gen_len-5, max_length=input_length+gen_len, temperature=payload["temperature"], repetition_penalty=payload["repetition_penalty"], top_k=payload["top_k"], top_p=payload["top_p"], do_sample=True, num_return_sequences=payload["batch_size"])

            #creating final answer
            final_answer={"text":{}, "truncated":truncated}

            #looping through batch
            for z in range(0, payload["batch_size"]):

                #getting just the generated text
                text_gen=self.tokenizer.decode(prediction[z][input_length:], skip_special_tokens=True)

                #parsing the generated text into sentences
                list_gen=nltk.sent_tokenize(text_gen)

                #cleaning up text
                final_answer["text"][z] = text_cleanup(text_gen, list_gen, payload["mode"])

        return final_answer
