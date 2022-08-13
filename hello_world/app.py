import sys

sys.path.append("/mnt/efs/recmovdeps")
import os
import json
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
from typing import Union, List, Dict
from constants import TRAIN_CONSTANTS

from predict import Recommender

print(f"""
THINGS INSIDE EFS: {os.listdir("/mnt/efs")}
{os.listdir("/mnt/efs/recmovdeps")}
""")

model_available = False
if os.path.exists("/mnt/efs/bert4rec-state-dict.pth"):
    model_available = True

if model_available:
    rec_obj = Recommender(TRAIN_CONSTANTS.MODEL_PATH)

app = FastAPI()


class RecommendRequestModel(BaseModel):
    id: Union[str, int, None] = None
    sequence: List[Union[str, int]]
    history: Union[int, None] = 10


@app.get("/")
def parent_get():
    return "Hello World"


@app.post("/api/recommend")
def api_recommend(payload: RecommendRequestModel):
    if model_available:
        try:
            id = payload.id
            sequence = payload.sequence
            sequence = [int(s) for s in sequence]
            history = payload.history
            seq, hist, rec = rec_obj.recommend(sequence=sequence,
                                               num_recs=history)
            return dict(ok=True,
                        recommendations=rec,
                        seq=seq,
                        hist=hist,
                        id=id,
                        message=dict(traceback="", message="success", info=""))
        except Exception as e:
            print(f'GOT EXCEPTION: {str(e)}')
            print(f'TRACEBACK: {traceback.format_exc()}')
            return dict(ok=False,
                        message=dict(traceback=f"""
                    TRACEBACK: {traceback.format_exc()}
                    ERROR: {str(e)}
                    """,
                                     message="failure",
                                     info="Error recommending"))
    else:
        return dict(ok=False,
                    message={
                        "traceback": "",
                        "message": "Model not available in efs",
                        "info": f"Things in EFS: {os.listdir('/mnt/efs')}"
                    })


lambda_handler = Mangum(app, lifespan="off")

# import requests

# def lambda_handler(event, context):
#     """Sample pure Lambda function

#     Parameters
#     ----------
#     event: dict, required
#         API Gateway Lambda Proxy Input Format

#         Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

#     context: object, required
#         Lambda Context runtime methods and attributes

#         Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

#     Returns
#     ------
#     API Gateway Lambda Proxy Output Format: dict

#         Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
#     """

#     # try:
#     #     ip = requests.get("http://checkip.amazonaws.com/")
#     # except requests.RequestException as e:
#     #     # Send some context about this error to Lambda Logs
#     #     print(e)

#     #     raise e

#     return {
#         "statusCode": 200,
#         "body": json.dumps({
#             "message": "hello world",
#             # "location": ip.text.replace("\n", "")
#         }),
#     }
