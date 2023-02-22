import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor

ENDPOINT =  "image-classification-2023-02-15-18-22-54-384"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a Predictor
    predictor = Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
   
    # Make a prediction:
    inferences = predictor.predict(image)
    event["body"]["inferences"] = inferences.decode('utf-8')
    
    # We return the data back to the Step Function    
    
    return {
        'statusCode': 200,
        'body': {
            "image_data": event["body"]["image_data"],
            "s3_bucket": event["body"]["s3_bucket"],
            "s3_key": event["body"]["s3_key"],
            "inferences": event["body"]["inferences"]
        }    
    }
