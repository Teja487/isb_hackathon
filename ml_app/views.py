from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"
image_index_name='image_index.html'
image_predict_name="image_predict.html"




im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
if torch.cuda.is_available():
    device = 'gpu'
else:
    device = 'cpu'

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length=60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        """
        for i,frame in enumerate(self.frame_extract(video_path)):
            if(i % a == first_frame):
                frames.append(self.transform(frame))
        """        
        # if(len(frames)<self.count):
        #   for i in range(self.count-len(frames)):
        #         frames.append(self.transform(frame))
        #print("no of frames", self.count)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype('uint8'))
    plt.show()


def predict(model,img,path = './', video_file_name=""):
  fmap,logits = model(img.to(device))
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)  
  return [int(prediction.item()),confidence]

def plot_heat_map(i, model, img, path = './', video_file_name=''):
  fmap,logits = model(img.to(device))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  #out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  result = heatmap * 0.5 + img*0.8*255
  # Saving heatmap - Start
  heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
  image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
  cv2.imwrite(image_name,result)
  # Saving heatmap - End
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
  return image_name

# Model Selection
def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass  # Handle cases where the filename format doesn't match expected

    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)  # Convert accuracy to float for proper comparison
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        print("No model found for the specified sequence length.")  # Handle no models found case

    return final_model

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    #print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else: 
        return False
ALLOWED_IMAGE_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'])

def allowed_image_file(filename):
    if (filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS):
        return True
    else: 
        return False

from django.http import HttpResponse   
def entry_page(request):
    if request.method == 'GET':
        # Render the entry page with two buttons
        return render(request, 'entry_page.html')  # Adjust your template name accordingly
    
    if request.method == 'POST':
        # Check which button was clicked
        action = request.POST.get('action')

        if action == 'video':
            # Redirect to the video upload page (index view)
            return redirect('ml_app:index')  # Make sure 'ml_app:index' matches your URL pattern for the index view
        
        elif action == 'image':  # You can replace 'another_action' with the other button action
            # Perform any action needed for the second button
            # For example, you can redirect to another view or handle another task
            return redirect('ml_app:image_index')  # Adjust the URL name accordingly

        else:
            # If neither button was clicked or an unknown action
            return HttpResponse("Unknown action", status=400)


def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3])
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name

        # Load validation dataset
        video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        # Load model
        if(device == "gpu"):
            model = Model(2).cuda()  # Adjust the model instantiation according to your model structure
        else:
            model = Model(2).cpu()  # Adjust the model instantiation according to your model structure
        model_name = os.path.join(settings.PROJECT_DIR, 'models', get_accurate_model(sequence_length))
        path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        start_time = time.time()
        # Display preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames: {len(frames)}")
        # Process each frame for preprocessing and face cropping
        padding = 40
        faces_found = 0
        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            # Face detection and cropping
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 0:
                continue

            top, right, bottom, left = face_locations[0]
            frame_face = frame[top - padding:bottom + padding, left - padding:right + padding]

            # Convert cropped face image to RGB and save
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # # No face detected
        # if faces_found == 0:
        #     return render(request, predict_template_name.html, {"no_faces": True})

        # Perform prediction
        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Prediction | ===>")
                prediction = predict(model, video_dataset[i], './', video_file_name_only)
                confidence = round(prediction[1], 1)
                output = "REAL" if prediction[0] == 1 else "FAKE"
                print("Prediction:", prediction[0], "==", output, "Confidence:", confidence)
                print("<=== | Prediction Done | ===>")
                print("--- %s seconds ---" % (time.time() - start_time))

                # Uncomment if you want to create heat map images
                # for j in range(sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))

            # Render results
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': heatmap_images,
                'original_video': production_video_name,
                'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
                'output': output,
                'confidence': confidence
            }

            if settings.DEBUG:
                return render(request, predict_template_name, context)
            else:
                return render(request, predict_template_name, context)

        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html')
        

from django.shortcuts import render, redirect
from .forms import ImageUploadForm  # Your form for handling image uploads
from django.conf import settings
import os
import shutil
import time
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from django.conf import settings
import os
import shutil
import time

def image_index(request):
    if request.method == 'GET':
        image_upload_form = ImageUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        return render(request, image_index_name, {"form": image_upload_form})
    else:
        image_upload_form = ImageUploadForm(request.POST, request.FILES)
        if image_upload_form.is_valid():
            image_file = image_upload_form.cleaned_data['upload_image_file']
            image_file_ext = image_file.name.split('.')[-1]
            image_content_type = image_file.content_type.split('/')[0]

            # Check if the file is an image (by MIME type)
            if image_content_type != 'image':
                image_upload_form.add_error("upload_image_file", "Only image files are allowed.")
                return render(request, image_index_name, {"form": image_upload_form})

            # Check file size limit
            if image_file.size > int(settings.MAX_UPLOAD_SIZE):
                image_upload_form.add_error("upload_image_file", "Maximum file size exceeded.")
                return render(request, image_index_name, {"form": image_upload_form})

            # Validate file extension (ensure it's a valid image format)
            if image_file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
                image_upload_form.add_error("upload_image_file", "Invalid image file type.")
                return render(request, image_index_name, {"form": image_upload_form})

            # Save the image file to the 'image_uploaded' folder using IMAGE_MEDIA_ROOT
            saved_image_file = 'uploaded_image_' + str(int(time.time())) + "." + image_file_ext
            #image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            image_upload_path = os.path.join(settings.PROJECT_DIR, 'image_uploaded', saved_image_file)

            # Save the file
            if settings.DEBUG:
                with open(image_upload_path, 'wb') as img_file:
                    shutil.copyfileobj(image_file, img_file)
                request.session['file_name'] = image_upload_path
            else:
                # In production mode, save the file to the appropriate directory
                #image_upload_path = os.path.join(settings.IMAGE_MEDIA_ROOT, 'app', 'image_uploaded', saved_image_file)
                image_upload_path = os.path.join(settings.PROJECT_DIR, 'app','image_uploaded', saved_image_file)
                with open(image_upload_path, 'wb') as img_file:
                    shutil.copyfileobj(image_file, img_file)
                request.session['file_name'] = image_upload_path

            return redirect('ml_app:image_predict')  # Redirect to prediction page
        else:
            return render(request, image_index_name, {"form": image_upload_form})


import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from django.shortcuts import render

# Define the media root for images
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build paths inside the project like this: os.path.join(PROJECT_DIR, ...)
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#IMAGE_MEDIA_ROOT = os.path.join(settings.BASE_DIR, 'IMAGE_UPLOADED')
IMAGE_MEDIA_ROOT = os.path.join(PROJECT_DIR, 'image_uploaded')

def load_model():
    # Load the model from the models directory4
    print("loading modelllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")
    model_json_path = os.path.join(settings.BASE_DIR, 'models', 'model.json')
    weights_path = os.path.join(settings.BASE_DIR, 'models', 'model.weights.h5')
    
    # Load JSON model architecture
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    
    # Load model and weights
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    
    print(" modelllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll loadedddddddddddddddddddddddddddddddddddddddds")
    
    return loaded_model

# Preprocess the image for prediction
# def preprocess_image(image_path):
#     print("entering preprocess functionnnnnnnnnnnnnnnnnnnnnn")
#     img = cv2.imread(image_path)
#     print("done readingggggggggggggggggggggggggggggggggggg")
#     img = cv2.resize(img, (224, 224)) 
#     print("dddddddddddddddddddddddddddddddddddddddddddd")# Resize for the model input
#     img = tf.keras.applications.efficientnet.preprocess_input(img)
#     return img
def preprocess_image(image_path):
    print("Entering preprocess function")
    img = cv2.imread(image_path)
    print(f"Done reading image: {image_path}")  # Print image path after reading
    img = cv2.resize(img, (224, 224)) 
    print("Resized image")  # Resize for the model input
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

# def image_predict(request):
#     if request.method == 'POST' and request.FILES.get('upload_image_file'):
#         # Handle file upload
#         print("************&&&&&&&&&&&&&&&&&&&&&&&&&&&&&**************")
#         upload_image_file = request.FILES['upload_image_file']
#         fs = FileSystemStorage(location=IMAGE_MEDIA_ROOT)  # Save in IMAGE_UPLOADED folder
#         filename = fs.save(upload_image_file.name, upload_image_file)
#         uploaded_image_path = os.path.join(IMAGE_MEDIA_ROOT, filename)
#         print("***********************************************************")

#         # Load the model
#         model = load_model()

#         # Preprocess the image
#         processed_image = preprocess_image(uploaded_image_path)
#         processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

#         # Make prediction
#         prediction = model.predict(processed_image)
#         predicted_class = "REAL" if prediction[0, 0] > 0.33 else "FAKE"

#         # Create context to render the result
#         context = {
#             'uploaded_image': os.path.join(settings.IMAGE_MEDIA_URLS, filename),
#             'predicted_class': predicted_class,
#             'confidence': prediction[0, 0]
#         }
#         if settings.DEBUG:
#                 return render(request, image_predict_name, context)
#         else:
#                 return render(request, image_predict_name, context)

def image_predict(request):
    if request.method == "GET":
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            print("************&&&&&&&&&&&&&&&&&&&&&&&&&&&&&**************")
            upload_image_file = request.session['file_name']
        upload_image_file = os.path.basename(upload_image_file)
        print(f"Uploaded image file path: {upload_image_file}")
        upload_image_file_only = os.path.splitext(upload_image_file)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', upload_image_file.split('/')[3])
            print("Production file name", production_video_name)
        else:
            production_video_name = upload_image_file

        # fs = FileSystemStorage(location=IMAGE_MEDIA_ROOT)  # Save in IMAGE_UPLOADED folder
        # filename = fs.save(upload_image_file.name, upload_image_file)
        
        #uploaded_image_path = os.path.join(settings.MEDIA_ROOT, upload_image_file)
        uploaded_image_path = os.path.join(settings.PROJECT_DIR, 'image_uploaded', upload_image_file)
        #uploaded_image_path = os.path.join(settings.PROJECTDIR, 'image_uploaded', upload_image_file)
        print(f"Updateeeeeeeeedddddddddddddddd :{uploaded_image_path}")
    
        # image_upload_path = os.path.join(settings.IMAGE_MEDIA_ROOT, saved_image_file)    
    # if request.method == 'GET' and request.FILES.get('upload_image_file'):
    #     # Handle file upload
    #     print("************&&&&&&&&&&&&&&&&&&&&&&&&&&&&&**************")
    #     upload_image_file = request.FILES['upload_image_file']
    #     fs = FileSystemStorage(location=IMAGE_MEDIA_ROOT)  # Save in IMAGE_UPLOADED folder
    #     filename = fs.save(upload_image_file.name, upload_image_file)
    #     uploaded_image_path = os.path.join(IMAGE_MEDIA_ROOT, filename)
    #     print("***********************************************************")

        # Load the model
        print("********************************8")
        model = load_model()
        print("model loading doneeeeeeeeeeeeeeeeeeeee")
        #image_upload_path = os.path.join(settings.IMAGE_MEDIA_ROOT, saved_image_file)

        # Preprocess the image
        processed_image = preprocess_image(uploaded_image_path)
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        print("doneeee with processinggggggggggggggggg")

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = "FAKE" if prediction[0, 0] > 0.5 else "REAL"
        print(f"predicted classssssssssss:{predicted_class}")
        print(f"confidenceeeeeeeeee:{prediction[0,0]}")
        #uploaded_image=os.path.join(settings.BASE_DIR, 'image_uploaded', upload_image_file)
        # Create context to render the 
 

# Assuming `upload_image_file` is the name of your uploaded image file
        uploaded_image = os.path.join(settings.MEDIA_URL, 'image_uploaded', upload_image_file)
# Make sure it looks something like: '/media/image_uploaded/your_image_file.jpg'

        context = {
            'uploaded_image': uploaded_image,
            'predicted_class': predicted_class,
            'confidence': prediction[0, 0]
        }

        # context = {
       
        #     'uploaded_image': uploaded_image,
        #     'predicted_class': predicted_class,
        #     'confidence': prediction[0, 0]
        # }
        print(context)
        return render(request, image_predict_name, context)  # Always return HttpResponse
    else:
        # Handle case where the method is not POST or file is not uploaded
        return render(request, '404.html')  # Adjust to the correct template


def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')
