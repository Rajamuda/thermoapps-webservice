# from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed
from django.conf import settings
from django.views.decorators.csrf import get_token, csrf_exempt

# Create your views here.
def index(request):
    data = {
        'status': True,
        'message': 'CSRF Token generator',
        'csrf_token': get_token(request)
    }
    return JsonResponse(data)

@csrf_exempt
def upload_image(request):
    import os, datetime, re, string

    if request.method != 'POST':
        return HttpResponseNotAllowed('Method Not Allowed')
    
    image = request.FILES['image']

    if image.content_type not in ['image/jpeg', 'image/png']:
        return JsonResponse({
            'status': False,
            'message': 'Hanya menerima file gambar dengan format JPEG atau PNG'
        })

    timestamp = str(int(datetime.datetime.timestamp(datetime.datetime.now())))
    
    upload_file_extension = image.content_type[6:]
    upload_file_name = os.path.splitext(image.name)[0].replace(" ", "_") + "_" + timestamp
    upload_file_folder = datetime.datetime.now().strftime('%Y.%m.%d')

    # create upload's directory if not exist based on current date
    if not os.path.exists(settings.UPLOAD_DIR / upload_file_folder):
        os.makedirs(settings.UPLOAD_DIR / upload_file_folder) 
    # relative path to upload's directory
    upload_location = upload_file_folder + "/" + upload_file_name + "." + upload_file_extension
    try:
        with open(settings.UPLOAD_DIR / upload_location, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
    except:
        return JsonResponse({
            'status': False,
            'message': 'File gagal diupload'
        })

    return JsonResponse({
        'status': True,
        'message': 'File berhasil diupload',
        'image': upload_location
    })

@csrf_exempt
def process(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed('Method Not Allowed')

    from .processors import NumpyEncoder, predict_img
    from PIL import Image
    import io, base64

    image_path = request.POST['image_path']

    model_1 = settings.KERAS_MODEL_DIR / '1_Model.042.h5'
    model_2 = settings.KERAS_MODEL_DIR / '2_Model.048.h5'
    model_3 = settings.KERAS_MODEL_DIR / '3_Model.038.h5'
    list_model = [model_1, model_2, model_3]
    
    try:
        img = Image.open(settings.UPLOAD_DIR / image_path)
    except FileNotFoundError as exception:
        return JsonResponse({
            'status': False,
            'message': 'File gambar tidak ditemukan'
        })

    img = img.resize((360, 360), Image.ANTIALIAS)

    X_pos, coor_pos, X_neg, coor_neg = predict_img(img, list_model, 30)

    # turn resized image to binary (bytes)
    img_bin = io.BytesIO()
    img.save(img_bin, format='PNG')
    img_bin = img_bin.getvalue()

    results = dict()
    results['status'] = True
    # "positif hamil" probability
    results['pos_prob'] = X_pos
    # "positif hamil" bounding box
    results['pos_bbox'] = coor_pos
    # "negatif hamil" probability
    results['neg_prob'] = X_neg
    # "negatif hamil" bounding box
    results['neg_bbox'] = coor_neg
    # bytes resized image
    results['image'] = {'type': 'image/png', 'base64_data': base64.b64encode(img_bin).decode('utf8')}

    return JsonResponse(results, encoder=NumpyEncoder)