import boto3
from botocore.session import Session
import json

# SSO 认证并获取临时凭证
sso_start_url = 'https://start.cn-north-1.home.awsapps.cn/directory/dps-china'
sso_region = 'cn-north-1'
sso_account_id = '579289528406'
sso_role_name = 'AWSPowerUserAccess'
region = 'cn-north-1'

# 初始化 SSO 客户端
sso_oidc = boto3.client('sso-oidc', region_name=sso_region)
sso = boto3.client('sso', region_name=sso_region)

# 注册客户端
register = sso_oidc.register_client(
    clientName='my-client',
    clientType='public'
)
client_id = register['clientId']
client_secret = register['clientSecret']

# 启动设备授权
auth = sso_oidc.start_device_authorization(
    clientId=client_id,
    clientSecret=client_secret,
    startUrl=sso_start_url
)

print(f"请在浏览器中打开以下链接完成认证：")
print(f"\n{auth['verificationUriComplete']}\n")
print(f"等待您在浏览器中完成登录...")

import time
import threading

def poll_token():
    while True:
        try:
            token = sso_oidc.create_token(
                clientId=client_id,
                clientSecret=client_secret,
                grantType='urn:ietf:params:oauth:grant-type:device_code',
                deviceCode=auth['deviceCode']
            )
            return token
        except Exception:
            time.sleep(5)

token = poll_token()
access_token = token['accessToken']

# 获取角色凭证
creds = sso.get_role_credentials(
    roleName=sso_role_name,
    accountId=sso_account_id,
    accessToken=access_token
)

aws_access_key = creds['roleCredentials']['accessKeyId']
aws_secret_key = creds['roleCredentials']['secretAccessKey']
session_token = creds['roleCredentials']['sessionToken']

# 用临时凭证查询 Athena
athena = boto3.client(
    'athena',
    region_name=region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    aws_session_token=session_token
)

# 执行 SQL 查询
response = athena.start_query_execution(
    QueryString="""
        SELECT DISTINCT equipmentnumber
        FROM data_cleansed."anyescalator"
        WHERE stepbandspeedleftavg IS NOT NULL
          AND stepbandspeedleftavg >= 0
          AND eventdate BETWEEN '2026-05-22' AND '2026-05-23'
        ORDER BY equipmentnumber
    """,
    ResultConfiguration={
        'OutputLocation': 's3://YOUR_QUERY_RESULTS_BUCKET/'
    }
)

query_id = response['QueryExecutionId']
print(f"查询已提交，ID: {query_id}")

# 等待查询完成
import time
while True:
    status = athena.get_query_execution(QueryExecutionId=query_id)
    state = status['QueryExecution']['Status']['State']
    if state == 'SUCCEEDED':
        break
    elif state in ('FAILED', 'CANCELLED'):
        print(f"查询失败: {state}")
        print(status['QueryExecution']['Status'].get('StateChangeReason'))
        exit(1)
    time.sleep(2)

# 获取结果
results = athena.get_query_results(QueryExecutionId=query_id)
for row in results['ResultSet']['Rows']:
    print(row['Data'][0]['VarCharValue'])