# Helper functions for visualizing the performance of the OCR model

from ctc_decoder import beam_search
import pandas as pd
from IPython.display import HTML
from io import BytesIO
import base64
import torch
import numpy as np

def convertLabelsToChars(labels, labelsToCharsDict):
    """converts an arrays of predicted labels into the corresponding text"""
    
    labels = np.array(labels)
    labels = labels[np.nonzero(labels)]
    string = "".join([labelsToCharsDict.get(i) for i in labels])
    
    #return " ".join([f'raw: {labels}', f'processed: {string}'])
    return string

def visualize_text(model, valGenerator, allChars, labelsToCharsDict, device, numExamples=2):
    """ visualize the results of using the model to perform OCR, without showing the images"""
    
    model.eval()
    images, X, targets, targetLengths, inputLengths = next(valGenerator)
    
    X = X.to(device)
    P = torch.exp(model(X)).detach()
    P = torch.roll(P, -1, 2) # need the probability of the blank character to be last to use ctcdecode
 
    print('Examples:')
    for index in range(numExamples):
        y = P[:,index,:].cpu()
        
        print((f'\tInput: {convertLabelsToChars(targets[index,:], labelsToCharsDict)}'
               f'\tPrediction: {beam_search(y, allChars)}'))
    
def visualize_image(model, valGenerator, allChars, labelsToCharsDict, device, numExamples=10):
    """ Visualize the output of the model on some example inputs, showing the images """
    # adapted from https://www.kaggle.com/code/stassl/displaying-inline-images-in-pandas-dataframe

    model.eval()
    images, X, targets, targetLengths, inputLengths = next(valGenerator)
    X = X.to(device)
    P = torch.exp(model(X)).detach()
    P = torch.roll(P, -1, 2) # need the probability of the blank character to be last to use ctcdecode

    r = pd.DataFrame()
    r['image'] = images[:numExamples]
    r['ground truth'] = [convertLabelsToChars(target, labelsToCharsDict) 
                         for target in targets[:numExamples]]
    r['recovered label'] = [beam_search(P[:, idx, :].cpu(), allChars) 
                            for idx in range(numExamples)]

    def image_base64(im):
        """encode an input image as a base64 string"""
    
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()
        
    def image_formatter(im):
        """convert images to encoded jpeg strings"""
    
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

    return HTML(r.to_html(formatters={'image': image_formatter}, escape=False))