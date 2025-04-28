# lambda/index.py
import json
import os
import boto3
import re  # 正規表現モジュールをインポート
from botocore.exceptions import ClientError
from urllib.request import urlopen, Request


# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値


# .envを読み込む関数（python-dotenvが使えないので作成）
def load_env_file(filepath=".env"):
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
# .envの読み込み実行
load_env_file()


# グローバル変数としてクライアントを初期化（初期値）
bedrock_client = None

# モデルID
#MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

def lambda_handler(event, context):
    try:
        # コンテキストから実行リージョンを取得し、クライアントを初期化
        global bedrock_client
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        #print("Using model:", MODEL_ID)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        

        # gemma用のリクエストペイロードを構築
        # 会話履歴を含める
        gemma_messages = ""
        for msg in messages:
            if msg["role"] == "user":
                gemma_messages += "<start_of_turn>user\n" + msg["content"] + "<end_of_turn>\n"
            elif msg["role"] == "assistant":
                gemma_messages += "<start_of_turn>model\n" + msg["content"] + "<end_of_turn>\n"
        gemma_messages += "<start_of_turn>model\n"
        
        
        # gemma用のリクエストペイロード
        url = os.environ.get("ENDPOINT_URL") + "/generate"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        data = {
            "prompt": gemma_messages,
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        # JSONエンコードしてbytesに変換
        json_data = json.dumps(data).encode('utf-8')
        
        # RequestにPOSTメソッドを明示的に指定
        req = Request(url, data=json_data, headers=headers, method="POST")

        try:
            with urlopen(req) as response:
                res = response.read().decode('utf-8')
                print(res)
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason}")
            print(e.read().decode("utf-8"))
        except urllib.error.URLError as e:
            print(f"URL Error: {e.reason}")


        # アシスタントの応答を取得
        assistant_response = json.loads(res)["generated_text"]
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
