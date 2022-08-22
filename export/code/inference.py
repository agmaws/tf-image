from tensorflow.keras.preprocessing import image
import numpy as np
import json
import sys
import requests
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import io

CLASSES = ["Priority", "Roundabout", "Signal"]

def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    img_data = json.loads(processed_input)['instances']
    return _process_output(response, context, img_data)


def _process_input(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    print(sys.getsizeof(data))
    
    if context.request_content_type == 'application/x-image':
        target_size=(224,224)
        img = Image.open(io.BytesIO(data.read()))
        img = img.convert('RGB')
        img = img.resize(target_size, Image.NEAREST)
        img = image.img_to_array(img)
        x = np.expand_dims(img, axis=0)

        return json.dumps({'instances': x.tolist()})
    else:
        _return_error(415, 'Unsupported content type in request "{}"'.format(context.request_content_type or 'Unknown'))


def _process_output(response, context, img_data):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    print("printing from op handler")

    results = json.loads(response.content)
    
    label_idx = np.argmax(results['predictions'][0])
    
    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode('utf-8'))
    response_content_type = 'application/x-image'
    img_array = np.array(img_data)
    img_buf = img_array[0,:,:,:]
    pil_image = Image.fromarray(img_buf.astype(np.uint8)).convert('RGB')

    draw = ImageDraw.Draw(pil_image)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.load_default()
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, 0),CLASSES[label_idx],(255,255,255),font=font)

    #img = image.img_to_array(img)
    with io.BytesIO() as output:
        pil_image.save(output, format="PNG")
        contents = output.getvalue()
    prediction = contents
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))