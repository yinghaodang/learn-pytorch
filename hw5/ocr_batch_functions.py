# Modified from https://github.com/kutvonenaki/simple_ocr
import numpy as np
from PIL import Image
import itertools
import torch

import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image
# Now PIL.Image.Resampling.BICUBIC is always recognized.

    
class OCR_generator:
    """Generator for the input data to the OCR model. We're also preparing 
    arrays for the CTC loss which are related to the output dimensions"""

    def __init__(self, baseGenerator, batchSize, char_to_lbl_dict,
                height, transform, epochSize=500, isValidation=False):
        """Inputs
        baseGenerator: the base trdg generator
        batchSize: number of examples fed to the NN simultaneously
        char_to_lbl_dict: mapping from character to its label (int number)
        height: we assume that the input here is already scaled to the correct height
        transform: transformer to augment the data set, the current base generator doesn'
                for example zoom, translate etc
        epochSize: number of images in an epoch
        isValidation: whether or not this is a validation data generator \
                (returns ground truth images if True) """
        self.base_generator = baseGenerator
        self.batch_size = batchSize
        self.char_to_lbl_dict = char_to_lbl_dict
        self.img_h = height
        self.epoch_size = epochSize
        self.transform = transform
        self.isValidation = isValidation
        
        # total number of unique characters
        self.num_chars = len(char_to_lbl_dict)

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch """
        return self.epoch_size


    def __next__(self):
        """Generate one batch of data"""

        # generate content for the batch as a list of lists
        generated_content = list(
            list(tup) for tup in itertools.islice(self.base_generator,0,self.batch_size))

        # preprocess the batch content
        originalImages, generated_content, img_w, max_word_len_batch = \
                self.preprocess_batch_imgs(generated_content)

        # allocate the vectors for batch labels (integers for each character in a word)
        # and the padded + preprocessed images
        targets = torch.full((self.batch_size, max_word_len_batch),0)
        batch_imgs = torch.zeros((self.batch_size, img_w, self.img_h, 3),dtype=torch.float32)

        # the length of the time axis in the output,
        # or equivalently the width of the image after convolutions. 
        # Needed to input in the CTC loss
        # each maxpooling halves the width dimension so in our model scaling is 1/4 with 2 maxpoolings
        timeSteps = int(img_w / 4)

        # the number of timeSteps output by the RNN
        targetLengths = torch.full((self.batch_size,), 0)

        # fill the batch
        for batch_ind in range(self.batch_size):

            # get the next new image and the word in it
            img_arr, word = generated_content[batch_ind]
            batch_imgs[batch_ind,:,:] = img_arr

            # pack the word into the target format needed for CTCLoss
            # each zero in the output corresponds to a special blank character
            # each other number is the index of a character: e.g. 1 is the first character 
            # in the vocabulary
            curLabels = torch.Tensor([self.char_to_lbl_dict[char] for char in word])
            targets[batch_ind, 0:len(curLabels)] = curLabels
            targetLengths[batch_ind] = len(word)

        inputLengths = torch.full((self.batch_size,), timeSteps)

        if self.isValidation:
            return originalImages, torch.transpose(batch_imgs, 1, 3), targets, targetLengths, inputLengths
        else:
            return torch.transpose(batch_imgs, 1, 3), targets, targetLengths, inputLengths

    def __iter__(self):
        return self
        
    def preprocess_batch_imgs(self,generated_content):
        """Function to do augmentations, pad images, return longest word len etc"""

        # check the largest image width and word len in the batch
        pil_images = [img for img, word in generated_content]
        max_width = max([img.size[0] for img in pil_images])
        max_word_len_batch = max([len(word) for img, word in generated_content])


        # expand img with to mod 4_ds so that the maxpoolings wil result into
        # well defined integer length for the mapped tdist dimension ("new width")
        if max_width % 4 == 0:
            img_w = max_width
        else:
            img_w = max_width + 4 - (max_width % 4)

        #augment batch images
        for batch_ind in range(self.batch_size):

            # pad the image width to the largest (fixed) image width
            pil_img = pil_images[batch_ind]
            width, height = pil_img.size

            new_img = Image.new(pil_img.mode, (img_w, self.img_h), (255,255,255))
            new_img.paste(pil_img, ((img_w - width) // 2, 0))

            #some additional augmentation
            img_tensor = self.transform(new_img)

            # save to batch, transpose because the "time axis" is width
            generated_content[batch_ind][0] = torch.transpose(img_tensor, 0, 2)

        return pil_images, generated_content, img_w, max_word_len_batch

