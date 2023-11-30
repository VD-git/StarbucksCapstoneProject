import json
import pandas as pd
import boto3

def lambda_handler(event, context):
    # TODO implement

    endpoint_name = 'sagemaker-scikit-learn-2023-11-30-22-09-57-712'

    json_body = event

    runtime = boto3.Session().client('sagemaker-runtime')

    response = runtime.invoke_endpoint(\
                    EndpointName = endpoint_name,\
                    ContentType = 'application/json',\
                    Accept = 'application/json',\
                    Body = pd.DataFrame(json_body, index = [0]).to_json())

    result = json.loads(response['Body'].read().decode('utf-8'))

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
