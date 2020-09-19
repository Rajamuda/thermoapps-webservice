# from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def index(request):
    data = {
        'status': True,
        'message': 'CSRF Token generator',
        'csrf_token': csrf.get_token(request)
    }
    return JsonResponse(data)

@csrf_exempt
def uploadImage(request):
    import os, time, datetime

    if request.method != 'POST':
        return HttpResponseNotAllowed('Method Not Allowed')
    # if request.method == 'GET':
    #     return HttpResponse(loader.get_template('index.html').render())
    
    image = request.FILES['image']

    if image.content_type not in ['image/jpeg', 'image/png']:
        return JsonResponse({
            'status': False,
            'message': 'Hanya menerima file gambar dengan format JPEG atau PNG'
        })

    timestamp = str(time.mktime(datetime.datetime.today().timetuple()))
    
    upload_file_extension = image.content_type[6:]
    upload_file_name = os.path.splitext(image.name)[0] + "_" + timestamp
    upload_file_folder = datetime.datetime.now().strftime('%Y.%m.%d')
    if not os.path.exists(upload_file_folder):
        os.makedirs(upload_file_folder) 
    
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

def process(request):
    from .processors import predict_img
    from PIL import Image
    import io

    image_path = request.GET['image_path']

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
    # "positif hamil" probability
    results['pos_prob'] = X_pos
    # "positif hamil" bounding box
    results['pos_bbox'] = coor_pos
    # "negatif hamil" probability
    results['neg_prob'] = X_neg
    # "negatif hamil" bounding box
    results['neg_bbox'] = coor_neg
    # bytes resized image
    # results['image'] = img_bin

    return HttpResponse(results)