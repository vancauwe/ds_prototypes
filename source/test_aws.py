import boto3

def list_buckets():
    s3_client = boto3.client('s3', 
                             #endpoint_url = "https://s3.amazonaws.com",
                             region_name = "eu-west-3"
                            )
    # parameters = s3_client.describe_parameters()['Parameters']
    # print(parameters)
    response = s3_client.list_buckets()
    print("These are the buckets")
    for b in response["Buckets"]:
        print(f' {b["Name"]}')

list_buckets()